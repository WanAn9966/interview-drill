"""Validate interview evaluation cases without contacting an AI provider."""

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / 'evals' / 'seeds.json'
CARD_KINDS = {'项目', '实习', '实习经历', '工作经历', '学历', '工作技能'}
LABELS = {'role_fit', 'question_grounding', 'answer_evidence',
          'feedback_grounding', 'actionability'}
PERSONAL_DATA = re.compile(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\d)1[3-9]\d{9}(?!\d)|sk-[A-Za-z0-9._-]{16,}')


def require(condition, case_id, message):
    if not condition:
        raise ValueError(f'{case_id}: {message}')


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def validate_case(case):
    require(isinstance(case, dict), 'case', 'must be an object')
    case_id = case.get('case_id', '<missing case_id>')
    require(nonempty(case.get('case_id')), case_id, 'case_id is required')
    require(type(case.get('schema_version')) is int and case['schema_version'] == 1,
            case_id, 'unsupported schema_version')
    for field in ('group_id', 'role_family', 'role', 'jd', 'answer'):
        require(nonempty(case.get(field)), case_id, f'{field} is required')
    require(nonempty(case.get('split')) and case['split'] in {'dev', 'holdout'},
            case_id, 'invalid split')

    source = case.get('source')
    require(isinstance(source, dict), case_id, 'source is required')
    require(nonempty(source.get('type')) and source['type'] in {'synthetic', 'authorized_deidentified'},
            case_id, 'invalid source type')
    require(nonempty(source.get('authorization_ref')), case_id, 'authorization_ref is required')
    require(source.get('deidentified') is True, case_id, 'source must be deidentified')
    if source['type'] == 'synthetic':
        require(source['authorization_ref'] == 'self-authored', case_id,
                'synthetic source must be self-authored')
    else:
        require(source['authorization_ref'].strip().lower() != 'self-authored', case_id,
                'real source needs a separate authorization record')

    cards = case.get('resume_cards')
    require(isinstance(cards, list) and bool(cards), case_id, 'resume_cards are required')
    card_by_id = {}
    for card in cards:
        require(isinstance(card, dict), case_id, 'invalid resume card')
        card_id = card.get('id')
        require(nonempty(card_id) and card_id not in card_by_id, case_id,
                'resume card ids must be unique and nonempty')
        require(nonempty(card.get('kind')) and card['kind'] in CARD_KINDS and nonempty(card.get('text')),
                case_id, 'resume card kind/text invalid')
        card_by_id[card_id] = card

    question = case.get('question')
    require(isinstance(question, dict), case_id, 'question is required')
    require(nonempty(question.get('text')), case_id, 'question text is required')
    require(nonempty(question.get('jd_quote')) and question['jd_quote'] in case['jd'],
            case_id, 'jd_quote is not in JD')
    require(nonempty(question.get('resume_card_id')), case_id, 'invalid resume_card_id')
    card = card_by_id.get(question['resume_card_id'])
    require(card is not None, case_id, 'question references an unknown resume card')
    require(nonempty(question.get('resume_quote')) and
            question['resume_quote'] in card['text'], case_id,
            'resume_quote is not in the referenced card')

    feedback = case.get('feedback_target')
    require(isinstance(feedback, dict), case_id, 'feedback_target is required')
    require(nonempty(feedback.get('answer_quote')) and
            feedback['answer_quote'] in case['answer'], case_id,
            'feedback quote is not in candidate answer')
    for field in ('gap', 'action'):
        require(nonempty(feedback.get(field)), case_id, f'feedback {field} is required')

    review = case.get('review')
    require(isinstance(review, dict), case_id, 'review is required')
    require(nonempty(review.get('status')) and review['status'] in {'pending', 'reviewed'},
            case_id, 'invalid review status')
    if review['status'] == 'pending':
        require(review.get('reviewer_id') is None and review.get('labels') is None and
                review.get('reviewed_on') is None, case_id,
                'pending case cannot claim human labels')
    else:
        require(nonempty(review.get('reviewer_id')), case_id, 'reviewer_id is required')
        require(isinstance(review.get('labels'), dict) and
                set(review['labels']) == LABELS and
                all(type(value) is int and 0 <= value <= 2
                    for value in review['labels'].values()), case_id,
                'reviewed case needs five 0-2 labels')
        try:
            date.fromisoformat(review.get('reviewed_on'))
        except (TypeError, ValueError):
            raise ValueError(f'{case_id}: reviewed_on must be an ISO date') from None

    require(PERSONAL_DATA.search(json.dumps(case, ensure_ascii=False)) is None,
            case_id, 'possible personal data or API key')


def validate_cases(cases):
    require(isinstance(cases, list) and bool(cases), 'dataset', 'nonempty array required')
    ids = set()
    group_splits = {}
    for case in cases:
        validate_case(case)
        case_id = case['case_id']
        require(case_id not in ids, case_id, 'duplicate case_id')
        ids.add(case_id)
        group = case['group_id']
        require(group not in group_splits or group_splits[group] == case['split'],
                case_id, 'group_id appears in both splits')
        group_splits[group] = case['split']
    return {'cases': len(cases), 'role_families': dict(sorted(Counter(
        case['role_family'] for case in cases).items())),
        'splits': dict(sorted(Counter(case['split'] for case in cases).items())),
        'review_status': dict(sorted(Counter(case['review']['status'] for case in cases).items()))}


def load_dataset(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result

    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', nargs='?', type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    cases = load_dataset(args.dataset)
    print(json.dumps(validate_cases(cases), ensure_ascii=True, sort_keys=True))


if __name__ == '__main__':
    main()
