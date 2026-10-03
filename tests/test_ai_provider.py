import base64
import io
import json
import os
import unittest
import wave
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

import ai_provider as provider
import app as server
import engine


def wav_sample():
    output = io.BytesIO()
    with wave.open(output, 'wb') as writer:
        writer.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        writer.writeframes(b'\0\0' * 240)
    return output.getvalue()


class BailianTransport(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            'AI_PROVIDER': 'bailian', 'BAILIAN_REGION': 'beijing',
            'DASHSCOPE_API_KEY': 'test-only-not-a-real-key'}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def transport(self, handler):
        client_type = httpx.AsyncClient
        return patch('httpx.AsyncClient', side_effect=lambda **kw:
                     client_type(transport=httpx.MockTransport(handler), **kw))

    async def test_chat_and_json_modes(self):
        requests = []
        def handler(req):
            self.assertEqual(req.url.host, 'dashscope.aliyuncs.com')
            self.assertEqual(req.headers['Authorization'], 'Bearer test-only-not-a-real-key')
            body = json.loads(req.content)
            requests.append(body)
            return httpx.Response(200, json={'choices': [{'message': {
                'content': '{"focus":"测试"}' if 'response_format' in body else '请说明你的贡献。'}}]})
        with self.transport(handler):
            self.assertIn('贡献', await engine.model_text('面试官', {}))
            self.assertEqual(await engine.model_text('返回JSON', {}, True), {'focus': '测试'})
        self.assertFalse(requests[0]['enable_thinking'])
        self.assertNotIn('response_format', requests[0])
        self.assertEqual(requests[1]['response_format'], {'type': 'json_object'})

    async def test_browser_audio_uses_base64_message(self):
        def handler(req):
            body = json.loads(req.content)
            self.assertEqual(body['model'], 'qwen3-asr-flash')
            self.assertNotIn('language', body['asr_options'])
            value = body['messages'][0]['content'][0]['input_audio']['data']
            self.assertEqual(value, 'data:audio/webm;base64,' + base64.b64encode(b'audio').decode())
            return httpx.Response(200, json={'choices': [{'message': {'content': '我负责 RAG 评估。'}}]})
        with self.transport(handler):
            self.assertEqual(await provider.transcribe_audio(b'audio', 'audio/webm', 'webm'), '我负责 RAG 评估。')

    async def test_long_speech_merges_all_chunks_without_forwarding_key(self):
        chunks = []
        def handler(req):
            if req.method == 'POST':
                body = json.loads(req.content)
                chunks.append(body['input']['text'])
                self.assertEqual(body['input']['voice'], 'Cherry')
                return httpx.Response(200, json={'output': {'audio': {
                    'url': 'http://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/test.wav?Signature=test'}}})
            self.assertEqual(req.url.scheme, 'https')
            self.assertNotIn('authorization', req.headers)
            return httpx.Response(200, content=wav_sample())
        text = '这是你的改进建议。' * 145
        with self.transport(handler):
            data, mime = await provider.synthesize(text)
        self.assertEqual(''.join(chunks), text)
        self.assertTrue(all(len(s) <= 600 for s in chunks))
        self.assertEqual(mime, 'audio/wav')
        with wave.open(io.BytesIO(data)) as reader:
            self.assertEqual(reader.getnframes(), 240 * len(chunks))

    async def test_generic_speech_protocol_preserved(self):
        os.environ.update(AI_PROVIDER='generic', STT_URL='https://example.test/stt',
                          STT_KEY='test', STT_MODEL='stt', TTS_URL='https://example.test/tts',
                          TTS_KEY='test', TTS_MODEL='tts')
        def handler(req):
            if req.url.path == '/stt':
                self.assertIn('multipart/form-data', req.headers['content-type'])
                return httpx.Response(200, json={'text': '回答'})
            self.assertEqual(json.loads(req.content)['response_format'], 'mp3')
            return httpx.Response(200, content=b'mp3')
        with self.transport(handler):
            self.assertEqual(await provider.transcribe_audio(b'audio', 'audio/webm', 'webm'), '回答')
            self.assertEqual(await provider.synthesize('问题'), (b'mp3', 'audio/mpeg'))

    async def test_missing_config_and_credentials_never_returned(self):
        client = TestClient(server.app)
        data = client.get('/api/config')
        self.assertTrue(data.json()['llm'])
        self.assertNotIn('test-only-not-a-real-key', data.text)
        del os.environ['DASHSCOPE_API_KEY']
        self.assertFalse(provider.configured('LLM'))
        self.assertEqual(provider.missing('STT'), ['DASHSCOPE_API_KEY'])
        os.environ['BAILIAN_REGION'] = 'unknown'
        self.assertIn('BAILIAN_REGION', provider.missing('LLM'))

    async def test_upstream_failure_is_redacted_and_bad_audio_rejected(self):
        def handler(req):
            return httpx.Response(401, json={'message': 'test-only-not-a-real-key'})
        with self.transport(handler):
            client = TestClient(server.app)
            response = client.post('/api/speech', json={'text': '问题'})
            self.assertEqual(response.status_code, 502)
            self.assertNotIn('test-only-not-a-real-key', response.text)
        for value in ['http://127.0.0.1/test.wav',
                      'https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com.evil.test/a']:
            with self.assertRaises(ValueError):
                provider.audio_url(value)
        with self.assertRaises(ValueError):
            provider.join_wav([b'not audio'])


if __name__ == '__main__':
    unittest.main()
