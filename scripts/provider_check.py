"""Credential-safe, bounded Beijing Bailian connectivity and speech round-trip probe."""

import argparse
import asyncio
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit
import wave

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ai_provider as provider
import engine

MODELS = {'LLM': 'qwen-plus', 'TTS': 'qwen3-tts-flash', 'STT': 'qwen3-asr-flash'}
PHRASE = '你好，请介绍你负责的项目。'
MAX_OUTPUT_TOKENS = 128
MAX_AUDIO_SECONDS = 30
# Public Beijing list prices checked 2026-10-04. Estimates, not cloud billing caps.
PRICE_DATE = '2026-10-04'
PRICES = {'input_per_million': 0.8, 'output_per_million': 2.0,
          'tts_per_10000_characters': 0.8, 'stt_per_second': 0.00022}
RESERVES = {'LLM': (4096 * 0.8 + MAX_OUTPUT_TOKENS * 2) / 1000000,
            'TTS': len(PHRASE) * 2 * 0.8 / 10000,
            'STT': MAX_AUDIO_SECONDS * 0.00022}


def preflight():
    """Do not display endpoint strings, key values, or arbitrary model strings."""
    results = {}
    for service, model in MODELS.items():
        cfg = provider.settings(service)
        reasons = []
        if not provider.bailian() or os.getenv('BAILIAN_REGION', '').strip() != 'beijing':
            reasons.append('requires_bailian_beijing')
        if not cfg['KEY']:
            reasons.append('missing_key')
        if cfg['MODEL'] != model:
            reasons.append('model_outside_probe_budget')
        path = ('/api/v1/services/aigc/multimodal-generation/generation' if service == 'TTS'
                else '/compatible-mode/v1/chat/completions')
        try:
            url = urlsplit(cfg['URL'])
            allowed_host = url.hostname == 'dashscope.aliyuncs.com' or re.fullmatch(
                r'[a-z0-9-]+\.cn-beijing\.maas\.aliyuncs\.com', url.hostname or '')
            valid = (allowed_host and url.scheme == 'https' and url.port in (None, 443)
                     and url.path == path and not url.query and not url.fragment
                     and not url.username and not url.password)
        except ValueError:
            valid = False
        if not valid:
            reasons.append('endpoint_not_beijing_official')
        results[service] = {'ready': not reasons, 'reasons': reasons,
                            'model': model if cfg['MODEL'] == model else 'unsupported'}
    return results


def failure_category(exc):
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return 'timeout'
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return {400: 'invalid_request', 401: 'authentication', 402: 'quota_or_billing',
                403: 'permission_or_quota', 404: 'model_or_endpoint',
                429: 'rate_limit_or_quota'}.get(status,
                    'provider_error' if status >= 500 else 'http_error')
    if isinstance(exc, httpx.RequestError):
        return 'network'
    return 'response_validation'


def normalized(text):
    return re.sub(r'[\W_]+', '', text, flags=re.UNICODE).lower()


async def probe_service(service, call, verify):
    responses = []
    started = time.perf_counter()
    result = {'service': service, 'model': MODELS[service], 'attempts': 1}
    value = None
    try:
        value = await asyncio.wait_for(call(responses.append), timeout=45)
        result.update(verify(value))
        result['status'] = 'ok'
    except Exception as exc:
        result.update(status='failed', category=failure_category(exc))
        # Never serialize exception strings: they may contain keys or signed URLs.
        if isinstance(exc, httpx.HTTPStatusError):
            result['failure_http_status'] = exc.response.status_code
        value = None
    result['elapsed_ms'] = round((time.perf_counter() - started) * 1000)
    result['responses'] = responses
    result['usage_status'] = 'reported' if responses and all(
        response['usage'] is not None for response in responses) else 'unknown'
    return result, value


def verify_question(value):
    if (not isinstance(value, dict) or value.get('speaker') != 'interviewer'
            or not engine.interviewer_question(value.get('question'))
            or not any(word in value['question'] for word in ('测试', '评估'))):
        raise ValueError('Question does not meet the synthetic interview checks')
    return {'interviewer_role': True, 'topic_keyword_check': True}


def verify_audio(value):
    data, mime = value
    if mime != 'audio/wav':
        raise ValueError('WAV required')
    with wave.open(io.BytesIO(data)) as reader:
        seconds = reader.getnframes() / reader.getframerate()
        if not 0 < seconds <= MAX_AUDIO_SECONDS or reader.getsampwidth() != 2:
            raise ValueError('Audio outside probe limits')
        frames = reader.readframes(reader.getnframes())
        if len(frames) != reader.getnframes() * reader.getnchannels() * 2 or not any(frames):
            raise ValueError('Empty or truncated audio')
        return {'audio_seconds': round(seconds, 3), 'sample_rate': reader.getframerate(),
                'channels': reader.getnchannels(), 'audio_bytes': len(data)}


def verify_transcript(value):
    if normalized(value) != normalized(PHRASE):
        raise ValueError('Synthetic speech round-trip mismatch')
    return {'normalized_round_trip_match': True}


def estimated_cost(results):
    total = 0
    for result in results:
        if result['status'] != 'ok' or result['usage_status'] != 'reported':
            return None
        usage = result['responses'][0]['usage']
        if result['service'] == 'LLM':
            if not {'prompt_tokens', 'completion_tokens'} <= usage.keys():
                return None
            total += (usage['prompt_tokens'] * 0.8 + usage['completion_tokens'] * 2) / 1000000
        elif result['service'] == 'TTS':
            if 'characters' not in usage:
                return None
            total += usage['characters'] * 0.8 / 10000
        else:
            if 'seconds' not in usage:
                return None
            total += usage['seconds'] * 0.00022
    return round(total, 8)


async def run_checks(*, live=False, budget_cny=0.05):
    checks = preflight()
    report = {'schema_version': 1, 'at_utc': datetime.now(timezone.utc).isoformat(),
              'live': live, 'preflight': checks,
              'budget_cny': budget_cny if math.isfinite(budget_cny) else None,
              'estimated_reservation_cny': round(sum(RESERVES.values()), 8),
              'price_checked_on': PRICE_DATE, 'prices': PRICES,
              'max_model_requests': 3, 'automatic_retries': 0, 'results': []}
    if not math.isfinite(budget_cny) or budget_cny < sum(RESERVES.values()):
        report.update(status='blocked', reason='budget_below_reservation')
        return report
    if not all(check['ready'] for check in checks.values()):
        report.update(status='blocked', reason='configuration')
        return report
    if not live:
        report['status'] = 'preflight_only'
        return report
    llm, _ = await probe_service('LLM', lambda observer: engine.model_text(
        '你是面试官，user是求职者。只围绕虚构项目与JD提一个具体追问，不代答。'
        '返回JSON：speaker固定为interviewer，question为问题。',
        {'role': '测试工程师', 'jd': '设计测试用例，说明评估依据。',
         'resume': '求职者在课程项目中负责接口测试。', 'answer': '我写了接口测试，但没有评估覆盖范围。'},
        json_mode=True, max_tokens=MAX_OUTPUT_TOKENS, observer=observer), verify_question)
    report['results'].append(llm)
    tts, audio = await probe_service('TTS', lambda observer: provider.synthesize(
        PHRASE, observer=observer), verify_audio)
    report['results'].append(tts)
    if audio is None:
        report['results'].append({'service': 'STT', 'model': MODELS['STT'],
                                  'attempts': 0, 'status': 'blocked',
                                  'category': 'tts_audio_unavailable', 'usage_status': 'unknown'})
    else:
        stt, _ = await probe_service('STT', lambda observer: provider.transcribe_audio(
            audio[0], 'audio/wav', 'wav', observer=observer), verify_transcript)
        report['results'].append(stt)
    report['status'] = 'ok' if all(item['status'] == 'ok' for item in report['results']) else 'failed'
    report['estimated_cost_cny'] = estimated_cost(report['results'])
    report['actual_charge_cny'] = None  # Only the provider's bill can confirm this.
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Make up to three paid requests, no retries')
    parser.add_argument('--budget-cny', type=float, default=0.05)
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / 'provider-check.json')
    args = parser.parse_args()
    provider.load_env_file(ROOT / '.env')
    report = asyncio.run(run_checks(live=args.live, budget_cny=args.budget_cny))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False) + '\n',
                           encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, allow_nan=False))
    return 0 if report['status'] in ('ok', 'preflight_only') else 1


if __name__ == '__main__':
    sys.exit(main())
