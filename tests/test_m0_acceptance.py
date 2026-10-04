import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_provider import load_env_file
from scripts.m0_acceptance import isolated_env, provider_evidence


class M0Acceptance(unittest.TestCase):
    def test_isolated_env_cannot_reload_real_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.env'
            path.write_text('AI_PROVIDER=bailian\nDASHSCOPE_API_KEY=private-test-key\n'
                            'LLM_KEY=private-test-key\nLLM_URL=https://example.invalid\n',
                            encoding='utf-8')
            env = isolated_env(Path(directory) / 'data')
            with patch.dict(os.environ, env, clear=True):
                load_env_file(path)
                self.assertEqual(os.environ['AI_PROVIDER'], 'generic')
                self.assertEqual(os.environ['DASHSCOPE_API_KEY'], '')
                self.assertEqual(os.environ['LLM_KEY'], '')
                self.assertEqual(os.environ['LLM_URL'], '')

    def test_partial_or_nonlive_provider_evidence_cannot_pass(self):
        data = {'live': True, 'status': 'ok', 'at_utc': '2026-10-04T06:32:08+00:00',
                'results': [{'service': service, 'status': 'ok', 'attempts': 1,
                             'usage_status': 'reported',
                             'responses': [{'http_status': 200, 'usage': {'total_tokens': 1}}]}
                            for service in ('LLM', 'TTS', 'STT')]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evidence.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            self.assertTrue(provider_evidence(path)['historical_only'])
            for key, value in (('live', False), ('status', 'failed'),
                               ('results', data['results'][:2])):
                with self.subTest(key=key):
                    path.write_text(json.dumps({**data, key: value}), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        provider_evidence(path)


if __name__ == '__main__':
    unittest.main()
