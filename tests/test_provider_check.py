import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import httpx

import ai_provider as provider
from scripts import provider_check as check


def speech_sample():
    output = io.BytesIO()
    with wave.open(output, 'wb') as writer:
        writer.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        writer.writeframes(b'\x01\x00' * 24000)
    return output.getvalue()


class ProviderDiagnostics(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'AI_PROVIDER': 'bailian', 'BAILIAN_REGION': 'beijing',
                                      'DASHSCOPE_API_KEY': 'test-private-value'}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def transport(self, handler):
        original = httpx.AsyncClient
        return patch('httpx.AsyncClient', side_effect=lambda **kw:
                     original(transport=httpx.MockTransport(handler), **kw))

    async def test_default_and_failed_preflight_never_send_requests(self):
        def fail(req):
            self.fail('Preflight made a network request')
        with self.transport(fail):
            self.assertEqual((await check.run_checks())['status'], 'preflight_only')
            self.assertEqual((await check.run_checks(live=True, budget_cny=0))['reason'],
                             'budget_below_reservation')
            for budget in (float('nan'), float('inf'), -1):
                report = await check.run_checks(live=True, budget_cny=budget)
                self.assertEqual(report['status'], 'blocked')
                json.dumps(report, allow_nan=False)
            os.environ['STT_URL'] = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions'
            self.assertEqual((await check.run_checks(live=True))['reason'], 'configuration')
            del os.environ['STT_URL']
            del os.environ['DASHSCOPE_API_KEY']
            self.assertFalse(check.preflight()['LLM']['ready'])

    async def test_actual_transports_round_trip_with_usage_and_output_limit(self):
        requests = []
        def handler(req):
            if req.method == 'GET':
                self.assertNotIn('authorization', req.headers)
                return httpx.Response(200, content=speech_sample())
            body = json.loads(req.content)
            requests.append(body['model'])
            if body['model'] == 'qwen-plus':
                self.assertEqual(body['max_tokens'], 128)
                return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({
                    'speaker': 'interviewer', 'question': '你如何评估接口测试的覆盖范围？'})}}],
                    'usage': {'prompt_tokens': 120, 'completion_tokens': 25,
                              'private_data': 'test-private-value'}})
            if body['model'] == 'qwen3-tts-flash':
                return httpx.Response(200, json={'output': {'audio': {'url':
                    'https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/test.wav?Signature=secret'}},
                    'usage': {'characters': 25}})
            audio = body['messages'][0]['content'][0]['input_audio']['data']
            self.assertTrue(audio.startswith('data:audio/wav;base64,'))
            return httpx.Response(200, json={'choices': [{'message': {'content': check.PHRASE}}],
                                            'usage': {'seconds': 1}})
        with self.transport(handler):
            report = await check.run_checks(live=True)
        self.assertEqual(report['status'], 'ok')
        self.assertEqual(requests, list(check.MODELS.values()))
        self.assertAlmostEqual(report['estimated_cost_cny'], 0.002366)
        self.assertIsNone(report['actual_charge_cny'])
        serialized = json.dumps(report)
        for private in ('test-private-value', 'Signature', check.PHRASE):
            self.assertNotIn(private, serialized)

    async def test_auth_failure_is_redacted_and_not_retried(self):
        requests = []
        def handler(req):
            requests.append(req)
            return httpx.Response(401, json={'message': 'test-private-value'})
        with self.transport(handler):
            report = await check.run_checks(live=True)
        self.assertEqual(len(requests), 2)  # LLM and TTS once; no audio to send to STT.
        self.assertEqual(report['results'][0]['category'], 'authentication')
        self.assertEqual(report['results'][2]['attempts'], 0)
        self.assertEqual(report['results'][2]['status'], 'blocked')
        self.assertIsNone(report['estimated_cost_cny'])
        self.assertNotIn('test-private-value', json.dumps(report))

    async def test_failure_classification_preserves_usage_after_validation_failure(self):
        request = httpx.Request('POST', 'https://example.invalid')
        for status, category in ((403, 'permission_or_quota'), (429, 'rate_limit_or_quota'),
                                 (500, 'provider_error'), (400, 'invalid_request')):
            with self.subTest(status=status):
                response = httpx.Response(status, request=request)
                error = httpx.HTTPStatusError('private', request=request, response=response)
                self.assertEqual(check.failure_category(error), category)
        self.assertEqual(check.failure_category(httpx.ReadTimeout('private')), 'timeout')
        self.assertEqual(check.failure_category(httpx.ConnectError('private')), 'network')
        async def call(observer):
            observer({'http_status': 200, 'usage': {'prompt_tokens': 100}})
            return {'speaker': 'candidate', 'question': '我负责测试。'}
        result, _ = await check.probe_service('LLM', call, check.verify_question)
        self.assertEqual(result['category'], 'response_validation')
        self.assertEqual(result['usage_status'], 'reported')

    def test_untrusted_endpoint_rejected_and_workspace_endpoint_supported(self):
        for url in ('http://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
                    'https://dashscope.aliyuncs.com.evil.test/compatible-mode/v1/chat/completions',
                    'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions?key=secret'):
            with self.subTest(url=url), patch.dict(os.environ, {'LLM_URL': url}):
                self.assertFalse(check.preflight()['LLM']['ready'])
        with patch.dict(os.environ, {'LLM_URL':
                'https://workspace-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions'}):
            self.assertTrue(check.preflight()['LLM']['ready'])

    def test_unknown_usage_is_not_zero_and_non_numeric_fields_are_dropped(self):
        records = []
        provider.provider_json(httpx.Response(200, request=httpx.Request('POST', 'https://example.invalid'),
            json={'usage': {'total_tokens': True, 'characters': 'private'}}), records.append)
        self.assertIsNone(records[0]['usage'])

    def test_local_env_has_no_database_side_effect_and_preserves_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            env_file = Path(folder) / '.env'
            env_file.write_text('\ufeff# comment\nBAILIAN_REGION=singapore\nTTS_VOICE="Cherry"\n',
                                encoding='utf-8')
            provider.load_env_file(env_file)
            self.assertEqual(os.environ['BAILIAN_REGION'], 'beijing')
            self.assertEqual(os.environ['TTS_VOICE'], 'Cherry')
            self.assertFalse((Path(folder) / 'data').exists())

    def test_round_trip_mismatch_and_long_audio_do_not_pass(self):
        self.assertEqual(check.verify_transcript('你好 请介绍你负责的项目'),
                         {'normalized_round_trip_match': True})
        with self.assertRaises(ValueError):
            check.verify_transcript('完全不同的识别结果')
        with self.assertRaises(ValueError):
            check.verify_audio((b'not wav', 'audio/mpeg'))
        output = io.BytesIO()
        with wave.open(output, 'wb') as writer:
            writer.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            writer.writeframes(b'\x01\x00' * 24000 * 31)
        with self.assertRaises(ValueError):
            check.verify_audio((output.getvalue(), 'audio/wav'))


if __name__ == '__main__':
    unittest.main()
