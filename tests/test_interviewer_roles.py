import copy
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import engine


class InterviewerRoles(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.state = {
            'mode': 'live', 'main_index': 1, 'depth': 1, 'style': '严格',
            'profile': {'name': '应用开发'}, 'jd': '实现检索与评估',
            'cards': [{'kind': '项目', 'text': '我负责知识库检索和评估。'}],
            'plan': [{'topic': '自我介绍', 'question': '请介绍自己。', 'anchor': ''},
                     {'topic': '项目深挖', 'question': '请说明个人贡献。', 'anchor': '我负责检索评估。'}],
            'messages': [{'role': 'assistant', 'text': '请说明你负责的工作。'},
                         {'role': 'user', 'text': '我负责检索评估，使用人工标注的问题对比方案。'}]}

    async def test_opening_never_impersonates_resume_owner(self):
        state = {**self.state, 'main_index': 0, 'depth': 0, 'messages': []}
        with patch.object(engine, 'model_text', AsyncMock()) as model:
            question = await engine.next_question(state)
        model.assert_not_awaited()
        self.assertIn('我是本次模拟面试的面试官', question)
        self.assertIn('请用一分钟介绍自己', question)

    async def test_candidate_answer_rejected_then_retried(self):
        bad = {'speaker': 'interviewer', 'question': '你好，我是张同学。我负责知识库评估。你想了解什么？'}
        good = {'speaker': 'interviewer', 'question': '你如何验证人工标注问题能代表实际用户的问题？'}
        with patch.object(engine, 'model_text', AsyncMock(side_effect=[bad, good])) as model:
            result = await engine.next_question(self.state)
        self.assertEqual(result, good['question'])
        self.assertEqual(model.await_count, 2)

    async def test_wrong_speaker_or_invalid_output_uses_question_fallback(self):
        for result in [{'speaker': 'candidate', 'question': '我负责接口开发。'},
                       {'speaker': 'interviewer', 'question': '下面给出参考答案。'},
                       '你好，我是一名应届生。']:
            with self.subTest(result=result):
                with patch.object(engine, 'model_text', AsyncMock(return_value=result)):
                    question = await engine.next_question(self.state)
                self.assertEqual(question, engine.fallback_question(self.state))
                self.assertTrue(engine.interviewer_question(question))

    async def test_quoted_candidate_statement_allowed(self):
        question = '你提到“我负责检索评估”，请说明评估集的构建依据。'
        self.assertTrue(engine.interviewer_question(question))
        with patch.object(engine, 'model_text', AsyncMock(return_value={
                'speaker': 'interviewer', 'question': question})) as model:
            self.assertEqual(await engine.next_question(self.state), question)
        self.assertEqual(model.await_count, 1)

    async def test_native_chat_roles_and_old_bad_output_not_replayed(self):
        state = copy.deepcopy(self.state)
        state['messages'].insert(0, {'role': 'assistant', 'text': '我是张同学，我负责项目开发。'})
        received = []
        client_type = httpx.AsyncClient
        def handler(req):
            received.append(json.loads(req.content))
            return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({
                'speaker': 'interviewer', 'question': '你如何设计评估集？'}, ensure_ascii=False)}}]})
        with patch.dict(os.environ, {'AI_PROVIDER': 'generic', 'LLM_URL': 'https://test.invalid/chat',
                                     'LLM_KEY': 'test', 'LLM_MODEL': 'test'}, clear=True):
            with patch('httpx.AsyncClient', side_effect=lambda **kw:
                       client_type(transport=httpx.MockTransport(handler), **kw)):
                self.assertEqual(await engine.next_question(state), '你如何设计评估集？')
        messages = received[0]['messages']
        self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'user', 'system'])
        self.assertEqual(messages[-2]['content'], self.state['messages'][-1]['text'])
        self.assertNotIn('我是张同学', str(messages))
        self.assertIn('user是求职者', messages[-1]['content'])


if __name__ == '__main__':
    unittest.main()
