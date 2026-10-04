import base64
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid
from contextlib import contextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import TRACKS, build_profile, build_report, configured, make_plan, next_question

from resume_parser import KINDS, ResumeError, extract_document, parse_sections
from ai_provider import bailian, missing, transcribe_audio, synthesize, load_env_file

ROOT = Path(__file__).resolve().parent
load_env_file(ROOT / '.env')
DATA = Path(os.getenv('INTERVIEW_DATA_DIR', str(ROOT / 'data')))
DATA.mkdir(parents=True, exist_ok=True)
DB = DATA / 'interviews.sqlite3'


@contextmanager
def connection():
    c = sqlite3.connect(DB)
    try:
        with c:
            yield c
    finally:
        c.close()


with connection() as c:
    c.execute('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, version INTEGER, body TEXT)')

app = FastAPI(title='InterviewDrill · 本地面试工作台')


@app.middleware('http')
async def local_guard(request, call_next):
    origin = request.headers.get('origin')
    if origin and origin != str(request.base_url).rstrip('/'):
        return Response('不允许跨站访问本地服务', status_code=403)
    if request.headers.get('sec-fetch-site') == 'cross-site':
        return Response(status_code=403)
    return await call_next(request)


def get_session(sid):
    with connection() as c:
        row = c.execute('SELECT body FROM sessions WHERE id=?', (sid,)).fetchone()
    if not row:
        raise HTTPException(404, '会话不存在')
    return json.loads(row[0])


def save(s, new=False):
    old = s['version']
    s['version'] += 1
    with connection() as c:
        if new:
            c.execute('INSERT INTO sessions VALUES (?,?,?)', (s['id'], s['version'], json.dumps(s, ensure_ascii=False)))
        else:
            changed = c.execute('UPDATE sessions SET version=?,body=? WHERE id=? AND version=?',
                                (s['version'], json.dumps(s, ensure_ascii=False), s['id'], old))
            if not changed.rowcount:
                raise HTTPException(409, '会话已更新，请刷新后继续')


def check_version(s, version):
    if s['version'] != version:
        raise HTTPException(409, '会话已更新，请刷新后继续')


def message(s, role, text):
    s['messages'].append({'id': uuid.uuid4().hex, 'role': role, 'text': text,
                          'time': time.time(), 'topic': s['plan'][s['main_index']]['topic'],
                          'anchor': s['plan'][s['main_index']]['anchor'], 'depth': s['depth']})


def tick(s):
    if s['status'] == 'active':
        now = time.time()
        s['elapsed'] += now - s['last_tick']
        s['last_tick'] = now


async def generate(s):
    try:
        text = await next_question(s)
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        s['error'] = '模型调用失败。当前记录已保存；检查 API 地址、密钥及模型后点击重试。'
    else:
        message(s, 'assistant', text)
        s['pending'] = False
        s['error'] = ''
    save(s)
    return s


class ResumeIn(BaseModel):
    name: str = Field(default='', max_length=200)
    content: str = Field(default='', max_length=14000000)
    text: str = Field(default='', max_length=60000)


class Card(BaseModel):
    id: str
    text: str = Field(min_length=1, max_length=12000)
    kind: str = '其他'
    confirmed: bool = False


class StartIn(BaseModel):
    cards: list[Card] = Field(min_length=1, max_length=100)
    track: str = 'custom'
    role: str = Field(default='', max_length=150)
    jd: str = Field(default='', max_length=8000)
    style: str = '严格'
    minutes: int = Field(default=20, ge=5, le=60)
    mode: str = 'demo'


class TurnIn(BaseModel):
    version: int
    text: str = Field(default='', max_length=8000)


@app.get('/api/config')
def config():
    return {'tracks': TRACKS, 'llm': configured('LLM'), 'stt': configured('STT'), 'tts': configured('TTS'),
            'missing': {p.lower(): missing(p) for p in ('LLM', 'STT', 'TTS')},
            'provider': 'bailian' if bailian() else 'generic', 'resume_parser': 'sections-v2'}


@app.post('/api/resume')
def parse_resume(body: ResumeIn):
    try:
        warnings = []
        text = body.text
        if body.content:
            blob = base64.b64decode(body.content, validate=True)
            text, warnings = extract_document(body.name, blob)
        result = parse_sections(text, warnings)
        result['source_format'] = Path(body.name).suffix.lower().lstrip('.') if body.content else 'text'
        return result
    except ResumeError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        raise HTTPException(400, '文件读取失败，请检查格式或密码；扫描PDF请先OCR，也可以粘贴正文')


@app.post('/api/sessions')
async def start(body: StartIn):
    if body.mode not in ('demo', 'live'):
        raise HTTPException(400, '无效面试配置')
    if body.mode == 'live' and not configured('LLM'):
        raise HTTPException(400, '请先配置 ' + '、'.join(missing('LLM')))
    cards = [c.model_dump() for c in body.cards]
    for c in cards:
        if c['kind'] == '实习':
            c['kind'] = '实习经历'
    if sum(len(c['text']) for c in cards) > 60000:
        raise HTTPException(400, '确认的简历内容不能超过60000字')
    if any(not c['confirmed'] or not c['text'].strip() or c['kind'] not in KINDS for c in cards):
        raise HTTPException(400, '请确认经历片段并选择类型')
    role = body.role.strip() or TRACKS.get(body.track, {}).get('name', '目标岗位（依据JD）')
    if not body.role.strip() and body.track not in TRACKS and not body.jd.strip():
        raise HTTPException(400, '请输入目标岗位名称或粘贴岗位JD')
    try:
        profile = await build_profile(role, body.jd, body.mode)
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise HTTPException(502, '岗位分析失败，请检查模型配置后重试，页面上的简历仍保留')
    s = {'id': uuid.uuid4().hex, 'version': 0, 'created': time.time(), 'track': body.track,
         'role': role, 'profile': profile,
         'jd': body.jd, 'style': body.style, 'mode': body.mode, 'minutes': body.minutes,
         'cards': cards, 'plan': make_plan(cards, profile), 'main_index': 0, 'depth': 0,
         'status': 'active', 'elapsed': 0, 'last_tick': time.time(), 'messages': [],
         'pending': True, 'error': '', 'report': None}
    save(s, new=True)
    return await generate(s)


@app.get('/api/sessions')
def history():
    with connection() as c:
        rows = c.execute('SELECT body FROM sessions ORDER BY rowid DESC').fetchall()
    return [{k: s.get(k) for k in ('id', 'created', 'track', 'role', 'status', 'mode')} for s in (json.loads(r[0]) for r in rows)]


@app.get('/api/sessions/{sid}')
def read(sid: str):
    return get_session(sid)


@app.post('/api/sessions/{sid}/answer')
async def answer(sid: str, body: TurnIn):
    s = get_session(sid)
    check_version(s, body.version)
    if s['status'] != 'active' or s['pending']:
        raise HTTPException(409, '当前不能回答，请继续或重试生成问题')
    if not body.text.strip():
        raise HTTPException(400, '请先输入或录制回答')
    tick(s)
    message(s, 'user', body.text.strip())
    if s['elapsed'] >= s['minutes'] * 60:
        s['status'] = 'ended'
    elif s['depth'] < s['plan'][s['main_index']]['follow_limit']:
        s['depth'] += 1
    elif s['main_index'] + 1 < len(s['plan']):
        s['main_index'] += 1
        s['depth'] = 0
    else:
        s['status'] = 'ended'
    s['pending'] = s['status'] == 'active'
    save(s)
    return await generate(s) if s['pending'] else s


@app.post('/api/sessions/{sid}/{action}')
async def action(sid: str, action: str, body: TurnIn):
    s = get_session(sid)
    check_version(s, body.version)
    if action == 'retry' and s['pending'] and s['status'] == 'active':
        save(s)  # Reserve this revision before the external call.
        return await generate(s)
    if action == 'pause' and s['status'] == 'active':
        tick(s)
        s['status'] = 'paused'
    elif action == 'resume' and s['status'] == 'paused':
        s['status'] = 'active'
        s['last_tick'] = time.time()
    elif action == 'end' and s['status'] in ('active', 'paused'):
        tick(s)
        s['status'] = 'ended'
        s['pending'] = False
        s['error'] = ''
    elif action == 'report' and s['status'] == 'ended':
        try:
            s['report'] = await build_report(s)
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            raise HTTPException(502, '复盘生成失败或引用校验未通过，转录已保存，可重试')
    else:
        raise HTTPException(409, '当前状态不支持此操作')
    save(s)
    return s


@app.delete('/api/sessions/{sid}')
def delete(sid: str):
    with connection() as c:
        c.execute('PRAGMA secure_delete=ON')
        c.execute('DELETE FROM sessions WHERE id=?', (sid,))
    return {'ok': True}


@app.get('/api/sessions/{sid}/export')
def export(sid: str, format: str = 'md'):
    s = get_session(sid)
    if format == 'json':
        return Response(json.dumps(s, ensure_ascii=False, indent=2), media_type='application/json',
                        headers={'Content-Disposition': f'attachment; filename="interview-{sid[:8]}.json"'})
    lines = ['# 面试记录', f'岗位：{s.get("role", s["track"])}', f'模式：{s["mode"]}']
    for m in s['messages']:
        lines += ['', f'## {"面试官" if m["role"] == "assistant" else "候选人"} · {m["topic"]}', m['text']]
    if s['report']:
        lines += ['', '# 复盘', s['report']['summary']]
        for item in s['report']['items']:
            lines += ['', f'## 回答 {item["answer_id"][:8]}', f'原话：{item["quote"]}',
                      f'做得好：{item["strength"]}', f'缺口：{item["gap"]}', f'改进：{item["action"]}']
        lines += ['', '## 下次练习'] + ['- ' + a for a in s['report']['actions']]
    return Response('\n'.join(lines), media_type='text/markdown; charset=utf-8',
                    headers={'Content-Disposition': f'attachment; filename="interview-{sid[:8]}.md"'})


@app.post('/api/transcribe')
async def transcribe(request: Request):
    if not configured('STT'):
        raise HTTPException(400, '未配置语音识别服务，请使用浏览器识别或文字输入')
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > (10 if bailian() else 16) * 1024 * 1024:
            raise HTTPException(413, '录音过长，请分段回答')
    mime = request.headers.get('content-type', 'audio/webm').split(';')[0]
    extensions = {'audio/webm': 'webm', 'audio/ogg': 'ogg', 'audio/mp4': 'm4a', 'audio/wav': 'wav'}
    if mime not in extensions or not content:
        raise HTTPException(400, '不支持的音频格式或空录音')
    if bailian() and mime == 'audio/mp4':
        raise HTTPException(400, '请使用 Chrome/Edge 的 WebM 录音，或上传 WAV/OGG 音频')
    try:
        text = await transcribe_audio(bytes(content), mime, extensions[mime])
        return {'text': text}
    except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError):
        raise HTTPException(502, '语音识别失败，录音仍在当前页面，可下载或重新识别')


class SpeechIn(BaseModel):
    text: str = Field(min_length=1, max_length=2400)


@app.post('/api/speech')
async def speech(body: SpeechIn):
    if not configured('TTS'):
        raise HTTPException(400, '未配置语音合成')
    try:
        audio, mime = await synthesize(body.text)
        return Response(audio, media_type=mime)
    except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError):
        raise HTTPException(502, '语音合成失败，请重播或阅读文字')


@app.get('/')
def index():
    return FileResponse(ROOT / 'static' / 'index.html')


app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
