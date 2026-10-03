import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import app as server
import engine


class InterviewFlow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = server.DB
        server.DB = Path(self.tmp.name) / 'test.sqlite3'
        with server.connection() as c:
            c.execute('CREATE TABLE sessions (id TEXT PRIMARY KEY, version INTEGER, body TEXT)')
        self.client = TestClient(server.app)

    def tearDown(self):
        server.DB = self.old
        self.tmp.cleanup()

    def start(self, track='agent'):
        r = self.client.post('/api/sessions', json={'track':track, 'cards':[
            {'id':'p1', 'text':'我主导知识库问答应用，负责检索评估和服务部署。提升效果的统计口径待确认。',
             'kind':'项目', 'confirmed':True},
            {'id':'i1', 'text':'实习期间参与接口开发，负责权限校验和错误处理。', 'kind':'实习', 'confirmed':True}]})
        self.assertEqual(r.status_code,200,r.text)
        return r.json()

    def action(self,s,action,**kw):
        return self.client.post(f'/api/sessions/{s["id"]}/{action}',json={'version':s['version'],**kw})

    def test_complete_interview_export_and_delete(self):
        s=self.start()
        for _ in range(30):
            if s['status']=='ended': break
            response=self.action(s,'answer',text='我负责接口契约和评估集构建，先定义验收标准，再对比方案；结果口径还需要核实。')
            self.assertEqual(response.status_code,200,response.text)
            s=response.json()
        self.assertEqual(s['status'],'ended')
        s=self.action(s,'report').json()
        self.assertTrue(s['report']['items'])
        self.assertTrue(all(x['score'] is None for x in s['report']['items']))
        md=self.client.get(f'/api/sessions/{s["id"]}/export').text
        self.assertIn('实习经历',md)
        self.assertIn('规则演示',md)
        self.assertEqual(self.client.delete('/api/sessions/'+s['id']).status_code,200)
        self.assertEqual(self.client.get('/api/sessions/'+s['id']).status_code,404)

    def test_failure_saves_answer_and_retry_does_not_duplicate(self):
        s=self.start()
        old=s.copy()
        with patch.object(server,'next_question',AsyncMock(side_effect=ValueError('bad model'))):
            s=self.action(s,'answer',text='我负责检索服务。').json()
        self.assertTrue(s['pending'])
        self.assertTrue(s['error'])
        self.assertEqual(len([m for m in s['messages'] if m['role']=='user']),1)
        self.assertEqual(self.action(old,'answer',text='我负责检索服务。').status_code,409)
        s=self.action(s,'retry').json()
        self.assertFalse(s['pending'])
        self.assertEqual(len([m for m in s['messages'] if m['role']=='user']),1)

    def test_pause_prevents_answer_and_preserves_duration(self):
        s=self.start()
        s=self.action(s,'pause').json()
        self.assertEqual(self.action(s,'answer',text='暂停时不允许提交').status_code,409)
        elapsed=s['elapsed']
        with patch.object(server.time,'time',return_value=s['last_tick']+500):
            s=self.action(s,'resume').json()
        self.assertEqual(s['elapsed'],elapsed)

    def test_resume_parse_and_unconfirmed_card_rejected(self):
        result=self.client.post('/api/resume',json={'text':'项目经历\n本人负责 AI 问答应用后端开发，完成测试和部署。\n实习经历\n实习期间负责接口联调与异常处理。'})
        self.assertEqual(result.status_code,200)
        cards=result.json()['cards']
        self.assertTrue(cards)
        self.assertFalse(cards[0]['confirmed'])
        self.assertEqual(self.client.post('/api/sessions',json={'cards':cards}).status_code,400)

    def test_track_specific_plan_and_ungrounded_report_rejected(self):
        ai=self.start('ai');agent=self.start('agent');full=self.start('fullstack')
        self.assertEqual(len({s['plan'][-2]['question'] for s in [ai,agent,full]}),3)
        s=self.action(ai,'answer',text='我负责接口测试。').json()
        s=self.action(s,'end').json()
        s['mode']='live';server.save(s)
        aid=next(m['id'] for m in s['messages'] if m['role']=='user')
        fake={'summary':'假报告','items':[{'answer_id':aid,'quote':'用户从未说过的数字提升90%'}]}
        with patch.object(engine,'model_text',AsyncMock(return_value=fake)):
            self.assertEqual(self.action(s,'report').status_code,502)
        stored=self.client.get('/api/sessions/'+s['id']).json()
        self.assertIsNone(stored['report'])
        self.assertTrue(stored['messages'])

    def test_cross_origin_write_rejected(self):
        r=self.client.post('/api/resume',json={'text':'不应处理'},headers={'Origin':'https://untrusted.example'})
        self.assertEqual(r.status_code,403)

    def test_arbitrary_nontechnical_role_and_jd(self):
        r=self.client.post('/api/sessions',json={'role':'用户运营', 'track':'custom',
            'jd':'负责用户分层运营，设计活动并分析留存数据。', 'cards':[
                {'id':'op1','text':'本人负责社群活动策划，整理问卷并与设计同学协作上线活动。',
                 'kind':'项目','confirmed':True}]})
        self.assertEqual(r.status_code,200,r.text)
        s=r.json()
        self.assertEqual(s['role'],'用户运营')
        self.assertIn('用户运营',s['messages'][0]['text'])
        self.assertIn('用户分层运营',s['plan'][-2]['question'])
        self.assertNotIn('数据库',s['plan'][-2]['question'])

    def test_structured_resume_can_start_and_answer(self):
        from test_resume_parser import SAMPLE
        parsed=self.client.post('/api/resume',json={'text':SAMPLE}).json()
        cards=[{**c,'confirmed':True} for c in parsed['cards']]
        response=self.client.post('/api/sessions',json={'role':'产品经理','jd':'负责用户访谈和需求分析','cards':cards})
        self.assertEqual(response.status_code,200,response.text)
        s=response.json()
        self.assertTrue({'项目','实习经历','学历','工作技能'} <= {c['kind'] for c in s['cards']})
        response=self.action(s,'answer',text='我曾负责需求梳理，并参与应用的设计和验证。')
        self.assertEqual(response.status_code,200,response.text)
        self.assertFalse(response.json()['pending'])


if __name__=='__main__': unittest.main()
