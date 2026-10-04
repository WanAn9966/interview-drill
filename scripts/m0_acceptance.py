"""Repeatable M0 verification in isolated, credential-free subprocesses."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.baseline_check import SOURCES, free_port, check_repository_boundaries, smoke
from scripts.validate_evals import DEFAULT_DATASET, load_dataset, validate_cases


def isolated_env(data_dir):
    env = {key: value for key, value in os.environ.items()
           if not key.endswith(('_KEY', '_TOKEN', '_SECRET')) and key != 'PYTHONPATH'}
    # Empty values prevent app.load_env_file's setdefault from loading real keys.
    env.update(AI_PROVIDER='generic', BAILIAN_REGION='', DASHSCOPE_API_KEY='',
               INTERVIEW_DATA_DIR=str(data_dir), PYTHONIOENCODING='utf-8')
    for service in ('LLM', 'STT', 'TTS'):
        for field in ('KEY', 'URL', 'MODEL'):
            env[f'{service}_{field}'] = ''
    return env


def launch(root, port):
    process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app:app',
                                '--host', '127.0.0.1', '--port', str(port)],
                               cwd=root, env=isolated_env(root / 'data'),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with httpx.Client(timeout=1, trust_env=False) as client:
            for _ in range(60):
                if process.poll() is not None:
                    raise RuntimeError('isolated_server_exited')
                try:
                    config = client.get(f'http://127.0.0.1:{port}/api/config')
                    config.raise_for_status()
                    if any(config.json()[name] for name in ('llm', 'stt', 'tts')):
                        raise RuntimeError('isolated_server_has_credentials')
                    return process
                except httpx.HTTPError:
                    time.sleep(0.1)
        raise RuntimeError('isolated_server_not_ready')
    except BaseException:
        stop(process)
        raise


def stop(process):
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@contextmanager
def isolated_server():
    with tempfile.TemporaryDirectory(prefix='interview-m0-') as directory:
        root = Path(directory)
        for source in SOURCES:
            target = root / source
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / source, target)
        port = free_port()
        process = launch(root, port)
        try:
            yield root, port, process
        finally:
            stop(process)


def restart_check():
    with isolated_server() as (root, port, first):
        base = f'http://127.0.0.1:{port}'
        with httpx.Client(base_url=base, timeout=5, trust_env=False) as client:
            response = client.post('/api/sessions', json={
                'mode': 'demo', 'role': '产品经理', 'jd': '用户研究与需求分析',
                'cards': [{'id': 'm0-p1', 'kind': '项目', 'confirmed': True,
                           'text': '虚构课程项目：整理用户访谈和需求清单。'}]})
            response.raise_for_status()
            session = response.json()
            response = client.post(f"/api/sessions/{session['id']}/answer", json={
                'version': session['version'], 'text': '我负责整理访谈记录，结果仍需验证。'})
            response.raise_for_status()
            saved = response.json()
        stop(first)
        second = launch(root, port)
        try:
            with httpx.Client(base_url=base, timeout=5, trust_env=False) as client:
                response = client.get(f"/api/sessions/{saved['id']}")
                response.raise_for_status()
                restored = response.json()
                if restored != saved:
                    raise RuntimeError('restart_changed_saved_session')
                response = client.get(f"/api/sessions/{saved['id']}/export?format=json")
                response.raise_for_status()
                if response.json() != saved:
                    raise RuntimeError('export_changed_saved_session')
                response = client.delete(f"/api/sessions/{saved['id']}")
                response.raise_for_status()
                if client.get(f"/api/sessions/{saved['id']}").status_code != 404:
                    raise RuntimeError('delete_did_not_remove_session')
        finally:
            stop(second)
    return {'acknowledged_answer_after_restart': True, 'json_export_matches': True,
            'delete_verified': True, 'scope': 'completed_demo_request_only'}


def documentation_check():
    log = ROOT / 'docs' / '开发日志.md'
    text = log.read_text(encoding='utf-8')
    for anchor in ('d01', 'd02', 'd03', 'd04', 'd05', 'issues'):
        if f'id="{anchor}"' not in text:
            raise ValueError('development_log_missing_section')
    links = 0
    for source in (ROOT / 'README.md', ROOT / '开发日计划.md', log):
        for target in re.findall(r'\]\(([^)]+)\)', source.read_text(encoding='utf-8')):
            if '://' in target:
                continue
            name, _, anchor = target.partition('#')
            destination = source.parent / name if name else source
            if not destination.exists():
                raise ValueError('broken_local_document_link')
            if anchor and destination.resolve() == log.resolve() and f'id="{anchor}"' not in text:
                raise ValueError('broken_development_log_anchor')
            links += 1
    return {'local_links_checked': links, 'development_days': 5}


def provider_evidence(path):
    """Summarize historical evidence without treating it as a fresh live test."""
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('live') is not True or data.get('status') != 'ok':
        raise ValueError('successful_live_evidence_required')
    stamp = datetime.fromisoformat(data['at_utc'])
    if stamp.tzinfo is None or stamp > datetime.now(timezone.utc):
        raise ValueError('invalid_evidence_timestamp')
    results = data.get('results', [])
    if len(results) != 3 or {item['service'] for item in results} != {'LLM', 'STT', 'TTS'}:
        raise ValueError('three_services_required')
    for item in results:
        if (item['status'] != 'ok' or item['attempts'] != 1 or
                item.get('usage_status') != 'reported' or not item.get('responses') or
                not all(response.get('http_status') == 200 and response.get('usage')
                        for response in item['responses'])):
            raise ValueError('incomplete_service_evidence')
    return {'at_utc': stamp.isoformat(), 'services': ['LLM', 'STT', 'TTS'],
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'historical_only': True, 'fresh_live_requests': 0}


def source_fingerprint():
    digest = hashlib.sha256()
    files = set()
    for pattern in ('*.py', 'scripts/*.py', 'tests/*.py', 'static/*', 'evals/*.json'):
        files.update(path for path in ROOT.glob(pattern) if path.is_file())
    for path in sorted(files):
        digest.update(path.relative_to(ROOT).as_posix().encode('utf-8') + b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(output, evidence):
    output.mkdir(parents=True, exist_ok=True)
    results = []
    def step(name, call):
        started = time.perf_counter()
        try:
            detail = call()
            result = {'name': name, 'status': 'passed', 'detail': detail}
        except Exception as exc:
            result = {'name': name, 'status': 'failed', 'error_type': type(exc).__name__}
        result['elapsed_ms'] = round((time.perf_counter() - started) * 1000)
        results.append(result)

    with tempfile.TemporaryDirectory(prefix='interview-m0-tests-') as directory:
        def unit_tests():
            completed = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                cwd=ROOT, env=isolated_env(Path(directory)), capture_output=True,
                encoding='utf-8', errors='replace', timeout=120)
            log = completed.stdout + completed.stderr
            (output / 'unittest.log').write_text(log, encoding='utf-8')
            count = re.search(r'Ran (\d+) tests?', log)
            if completed.returncode or not count:
                raise RuntimeError('unit_tests_failed_see_unittest_log')
            return {'tests': int(count[1]), 'isolated_database': True}
        step('D01-D05_unit_regression', unit_tests)
    step('D01_repository_boundaries', check_repository_boundaries)
    def clean_flow():
        with isolated_server() as (_, port, __):
            return smoke(f'http://127.0.0.1:{port}')
    step('D01-D02_clean_business_flow', clean_flow)
    step('D02_saved_session_restart', restart_check)
    step('D03_evaluation_dataset', lambda: validate_cases(load_dataset(DEFAULT_DATASET)))
    step('D04_historical_live_evidence', lambda: provider_evidence(evidence))
    step('D05_documentation', documentation_check)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    report = {'schema_version': 1, 'at_utc': datetime.now(timezone.utc).isoformat(),
              'base_commit': head.stdout.strip(), 'source_sha256': source_fingerprint(),
              'python': sys.version.split()[0],
              'dependencies': {name: version(name) for name in ('fastapi', 'httpx', 'pypdf', 'python-docx')},
              'status': 'passed' if all(r['status'] == 'passed' for r in results) else 'failed',
              'results': results, 'fresh_live_requests': 0, 'production_release_approved': False}
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True))
    return 0 if report['status'] == 'passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / 'm0-acceptance')
    parser.add_argument('--provider-evidence', type=Path, default=ROOT / 'output' / 'provider-check.json')
    parser.add_argument('--serve', action='store_true', help='Serve an isolated UI until Enter is pressed')
    args = parser.parse_args()
    if args.serve:
        with isolated_server() as (_, port, __):
            print(f'Isolated credential-free UI: http://127.0.0.1:{port}', flush=True)
            input('Press Enter after browser verification to stop and clean up.\n')
        return 0
    return run(args.output, args.provider_evidence)


if __name__ == '__main__':
    sys.exit(main())
