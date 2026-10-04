"""Provider configuration and speech transports. Credentials stay on the server."""
import base64
import io
import math
import os
import re
import wave
from urllib.parse import urlsplit, urlunsplit

import httpx


def load_env_file(path):
    """Read the local configuration without importing the database or web app."""
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def provider_json(response, observer=None):
    """Expose only allowlisted numeric usage, never response text or signed URLs."""
    try:
        data = response.json()
    except ValueError:
        data = None
    if observer is not None:
        raw = data.get('usage') if isinstance(data, dict) else None
        usage = {}
        if isinstance(raw, dict):
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens',
                        'input_tokens', 'output_tokens', 'characters', 'seconds'):
                value = raw.get(key)
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    usage[key] = value
        observer({'http_status': response.status_code, 'usage': usage or None})
    response.raise_for_status()
    if not isinstance(data, dict):
        raise ValueError('服务返回的 JSON 结构无效')
    return data


def bailian():
    return os.getenv('AI_PROVIDER', 'generic').strip().lower() == 'bailian'


def settings(prefix):
    defaults = {}
    if bailian():
        region = os.getenv('BAILIAN_REGION', '').strip()
        host = {'beijing': 'dashscope.aliyuncs.com',
                'singapore': 'dashscope-intl.aliyuncs.com'}.get(region)
        defaults = {'KEY': os.getenv('DASHSCOPE_API_KEY', '').strip(),
                    'MODEL': {'LLM': 'qwen-plus', 'STT': 'qwen3-asr-flash',
                              'TTS': 'qwen3-tts-flash'}[prefix]}
        if host:
            path = ('/api/v1/services/aigc/multimodal-generation/generation' if prefix == 'TTS'
                    else '/compatible-mode/v1/chat/completions')
            defaults['URL'] = 'https://' + host + path
    return {k: os.getenv(f'{prefix}_{k}', '').strip() or defaults.get(k, '')
            for k in ('URL', 'KEY', 'MODEL')}


def missing(prefix):
    cfg = settings(prefix)
    result = []
    for k in ('URL', 'KEY', 'MODEL'):
        if not cfg[k]:
            result.append(('DASHSCOPE_API_KEY' if k == 'KEY' else 'BAILIAN_REGION')
                          if bailian() and k in ('KEY', 'URL') else f'{prefix}_{k}')
    return result


def configured(prefix):
    return not missing(prefix)


def headers(cfg):
    return {'Authorization': 'Bearer ' + cfg['KEY']}


async def transcribe_audio(content, mime, extension, *, observer=None):
    cfg = settings('STT')
    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
        if bailian():
            if mime == 'audio/mp4':
                raise ValueError('百炼识别请使用 WebM、OGG 或 WAV 录音')
            audio = 'data:' + mime + ';base64,' + base64.b64encode(content).decode('ascii')
            response = await client.post(cfg['URL'], headers=headers(cfg), json={
                'model': cfg['MODEL'], 'stream': False,
                'messages': [{'role': 'user', 'content': [
                    {'type': 'input_audio', 'input_audio': {'data': audio}}]}],
                'asr_options': {'enable_itn': False}})
        else:
            response = await client.post(cfg['URL'], headers=headers(cfg),
                data={'model': cfg['MODEL'], 'language': 'zh'},
                files={'file': (f'answer.{extension}', content, mime)})
        data = provider_json(response, observer)
        text = data['choices'][0]['message']['content'] if bailian() else data['text']
        if not isinstance(text, str) or not text.strip():
            raise ValueError('没有识别到有效语音')
        return text.strip()


def speech_chunks(text, limit=600):
    """Split near sentence boundaries without dropping text."""
    chunks = []
    while len(text) > limit:
        endings = list(re.finditer(r'[。！？!?；;\n]', text[:limit]))
        end = endings[-1].end() if endings else limit
        chunks.append(text[:end])
        text = text[end:]
    if text:
        chunks.append(text)
    return chunks


def audio_url(value):
    """Only download provider-owned OSS results; never forward API credentials."""
    parts = urlsplit(value)
    if (parts.scheme not in ('http', 'https') or parts.username or parts.password
            or parts.port not in (None, 80, 443)
            or not re.fullmatch(r'dashscope-result-[a-z0-9-]+\.oss-[a-z0-9-]+\.aliyuncs\.com',
                                parts.hostname or '')):
        raise ValueError('语音服务返回无效音频地址')
    return urlunsplit(('https', parts.hostname, parts.path, parts.query, ''))


def join_wav(parts):
    output = io.BytesIO()
    expected = None
    try:
        with wave.open(output, 'wb') as writer:
            for part in parts:
                with wave.open(io.BytesIO(part), 'rb') as reader:
                    fmt = (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(),
                           reader.getcomptype())
                    if expected is None:
                        expected = fmt
                        writer.setparams(reader.getparams())
                    elif expected != fmt:
                        raise ValueError('音频片段格式不一致')
                    writer.writeframes(reader.readframes(reader.getnframes()))
    except (wave.Error, EOFError) as exc:
        raise ValueError('语音服务没有返回有效 WAV 音频') from exc
    return output.getvalue()


async def synthesize(text, *, observer=None):
    cfg = settings('TTS')
    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
        if not bailian():
            response = await client.post(cfg['URL'], headers=headers(cfg), json={
                'model': cfg['MODEL'], 'input': text,
                'voice': os.getenv('TTS_VOICE', ''), 'response_format': 'mp3'})
            response.raise_for_status()
            return response.content, 'audio/mpeg'
        parts = []
        for chunk in speech_chunks(text):
            response = await client.post(cfg['URL'], headers=headers(cfg), json={
                'model': cfg['MODEL'], 'input': {'text': chunk,
                'voice': os.getenv('TTS_VOICE', '').strip() or 'Cherry', 'language_type': 'Chinese'}})
            data = provider_json(response, observer)
            url = audio_url(data['output']['audio']['url'])
            content = bytearray()
            async with client.stream('GET', url) as audio:
                audio.raise_for_status()
                async for block in audio.aiter_bytes():
                    content.extend(block)
                    if len(content) > 32 * 1024 * 1024:
                        raise ValueError('合成音频过大')
            parts.append(bytes(content))
        return join_wav(parts), 'audio/wav'
