import copy
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

import app as server
from engine import interviewer_question
from scripts.validate_evals import DEFAULT_DATASET, load_dataset, validate_cases


class EvaluationDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_dataset(DEFAULT_DATASET)

    def test_seed_business_chain_and_role_coverage(self):
        summary = validate_cases(self.cases)
        self.assertEqual(summary['cases'], 7)
        self.assertGreaterEqual(len(summary['role_families']), 5)
        self.assertEqual(summary['review_status'], {'pending': 7})
        kinds = {card['kind'] for case in self.cases for card in case['resume_cards']}
        self.assertTrue({'项目', '实习经历', '学历', '工作技能'} <= kinds)
        self.assertTrue(all(interviewer_question(case['question']['text']) for case in self.cases))

    def test_forged_evidence_and_missing_authorization_rejected(self):
        for mutate in (
            lambda case: case['question'].update(jd_quote='JD中不存在的要求'),
            lambda case: case['question'].update(resume_quote='简历中不存在的经历'),
            lambda case: case['feedback_target'].update(answer_quote='求职者没说过的话'),
            lambda case: case['source'].update(authorization_ref=''),
        ):
            with self.subTest(mutate=mutate):
                cases = copy.deepcopy(self.cases)
                mutate(cases[0])
                with self.assertRaises(ValueError):
                    validate_cases(cases)

    def test_pending_cannot_claim_human_review_and_pii_rejected(self):
        cases = copy.deepcopy(self.cases)
        cases[0]['review']['labels'] = {'role_fit': 2}
        with self.assertRaisesRegex(ValueError, 'pending case'):
            validate_cases(cases)
        cases = copy.deepcopy(self.cases)
        cases[0]['answer'] += ' 联系方式：example@example.com'
        with self.assertRaisesRegex(ValueError, 'personal data'):
            validate_cases(cases)

    def test_duplicate_case_and_group_split_rejected(self):
        cases = copy.deepcopy(self.cases)
        cases.append(copy.deepcopy(cases[0]))
        with self.assertRaisesRegex(ValueError, 'duplicate case_id'):
            validate_cases(cases)
        cases[-1]['case_id'] = 'another-case'
        cases[-1]['split'] = 'holdout'
        with self.assertRaisesRegex(ValueError, 'both splits'):
            validate_cases(cases)

    def test_duplicate_json_key_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'duplicate.json'
            path.write_text('[{"case_id":"a","case_id":"b"}]', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
                load_dataset(path)

    def test_each_seed_runs_through_interview_and_review(self):
        with tempfile.TemporaryDirectory() as directory:
            previous_db = server.DB
            server.DB = Path(directory) / 'eval.sqlite3'
            try:
                with server.connection() as connection:
                    connection.execute('CREATE TABLE sessions (id TEXT PRIMARY KEY, version INTEGER, body TEXT)')
                with TestClient(server.app) as client:
                    for case in self.cases:
                        with self.subTest(case_id=case['case_id']):
                            cards = [{**card, 'confirmed': True} for card in case['resume_cards']]
                            started = client.post('/api/sessions', json={
                                'mode': 'demo', 'role': case['role'], 'jd': case['jd'],
                                'cards': cards})
                            self.assertEqual(started.status_code, 200, started.text)
                            session = started.json()
                            self.assertEqual(session['profile']['name'], case['role'])
                            self.assertIn(case['question']['jd_quote'], session['jd'])
                            self.assertTrue(interviewer_question(session['messages'][0]['text']))
                            self.assertIn(case['role'], session['plan'][0]['question'])
                            answered = client.post(f"/api/sessions/{session['id']}/answer", json={
                                'version': session['version'], 'text': case['answer']})
                            self.assertEqual(answered.status_code, 200, answered.text)
                            session = answered.json()
                            self.assertEqual([message['role'] for message in session['messages'][:2]],
                                             ['assistant', 'user'])
                            self.assertEqual(session['messages'][1]['text'], case['answer'])
                            ended = client.post(f"/api/sessions/{session['id']}/end", json={
                                'version': session['version']})
                            self.assertEqual(ended.status_code, 200, ended.text)
                            session = ended.json()
                            reviewed = client.post(f"/api/sessions/{session['id']}/report", json={
                                'version': session['version']})
                            self.assertEqual(reviewed.status_code, 200, reviewed.text)
                            report = reviewed.json()['report']
                            self.assertIn(report['items'][0]['quote'], case['answer'])
                            self.assertIsNone(report['items'][0]['score'])
            finally:
                server.DB = previous_db


if __name__ == '__main__':
    unittest.main()
