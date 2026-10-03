"""Day-1, credential-free smoke test of a clean source checkout."""

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import httpx


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('app.py', 'engine.py', 'ai_provider.py', 'resume_parser.py',
           'static/index.html', 'static/app.js')


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def ignored(path):
    result = subprocess.run(['git', 'check-ignore', '-q', path], cwd=ROOT,
                            capture_output=True, check=False)
    return result.returncode == 0


def check_repository_boundaries():
    for path in ('.env', 'data/interviews.sqlite3', 'output/baseline.json',
                 '.playwright-cli/browser.yml'):
        if not ignored(path):
            raise RuntimeError(f'{path} is not excluded by .gitignore')
    tracked = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT,
                             capture_output=True, check=True).stdout.decode('utf-8').split('\0')
    for path in tracked:
        normalized = path.replace('\\', '/')
        if normalized in ('.env',) or normalized.startswith(('data/', 'output/', '.playwright-cli/')):
            raise RuntimeError('Private runtime data is tracked by Git')


def smoke(base):
    with httpx.Client(base_url=base, timeout=5, trust_env=False) as client:
        config = client.get('/api/config')
        config.raise_for_status()
        if config.json()['llm'] or config.json()['stt'] or config.json()['tts']:
            raise RuntimeError('Clean checkout unexpectedly has provider credentials')
        page = client.get('/')
        page.raise_for_status()
        if 'text/html' not in page.headers.get('content-type', ''):
            raise RuntimeError('Home page did not return HTML')
        resume = client.post('/api/resume', json={'text':
            '项目经历\n练习项目：负责需求整理与接口测试。\n实习经历\n参与团队协作与问题跟进。'})
        resume.raise_for_status()
        cards = [{**card, 'confirmed': True} for card in resume.json()['cards']]
        if not cards:
            raise RuntimeError('Resume did not produce evidence cards')
        started = client.post('/api/sessions', json={'mode': 'demo', 'role': '产品助理',
                                                     'cards': cards})
        started.raise_for_status()
        session = started.json()
        if not session['messages'] or session['messages'][0]['role'] != 'assistant':
            raise RuntimeError('Interview did not open with an interviewer question')
        answered = client.post(f"/api/sessions/{session['id']}/answer", json={
            'version': session['version'], 'text': '我负责整理需求和测试接口，结果仍需验证。'})
        answered.raise_for_status()
        session = answered.json()
        ended = client.post(f"/api/sessions/{session['id']}/end", json={
            'version': session['version']})
        ended.raise_for_status()
        session = ended.json()
        report = client.post(f"/api/sessions/{session['id']}/report", json={
            'version': session['version']})
        report.raise_for_status()
        if not report.json().get('report', {}).get('items'):
            raise RuntimeError('Demo report is empty')
        export = client.get(f"/api/sessions/{session['id']}/export")
        export.raise_for_status()
        if '面试记录' not in export.text:
            raise RuntimeError('Markdown export is incomplete')
        deleted = client.delete(f"/api/sessions/{session['id']}")
        deleted.raise_for_status()
    return {'home': 'ok', 'config_without_key': 'ok', 'resume': 'ok',
            'demo_interview': 'ok', 'report': 'ok', 'export': 'ok', 'delete': 'ok'}


def main():
    check_repository_boundaries()
    with tempfile.TemporaryDirectory(prefix='interview-baseline-') as scratch:
        root = Path(scratch)
        for source in SOURCES:
            destination = root / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / source, destination)
        env = {key: value for key, value in os.environ.items()
               if not key.endswith(('_KEY', '_TOKEN', '_SECRET'))
               and key not in ('PYTHONPATH', 'AI_PROVIDER')}
        env['AI_PROVIDER'] = 'generic'
        env['INTERVIEW_DATA_DIR'] = str(root / 'data')
        port = free_port()
        process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app:app',
                                    '--host', '127.0.0.1', '--port', str(port)],
                                   cwd=root, env=env, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        try:
            base = f'http://127.0.0.1:{port}'
            for _ in range(60):
                if process.poll() is not None:
                    raise RuntimeError('Clean server exited before becoming ready')
                try:
                    with httpx.Client(timeout=1, trust_env=False) as client:
                        if client.get(base + '/api/config').status_code == 200:
                            break
                except httpx.HTTPError:
                    time.sleep(0.2)
            else:
                raise RuntimeError('Clean server did not become ready')
            result = smoke(base)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    print(json.dumps({'credential_free_clean_checkout': 'ok',
                      'git_excludes_runtime_data': 'ok', **result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
