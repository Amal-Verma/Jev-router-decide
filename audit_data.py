import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


QUEUES = {'billing', 'technical', 'account', 'shipping', 'general'}
TEXT_FIELDS = ('subject', 'body', 'customer_tier', 'draft_reply', 'kb_article')
STATE_FIELDS = (*TEXT_FIELDS, 'amount')
LABEL_FIELDS = ('route', 'approve', 'grounded')


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def duplicate_summary(counts):
    repeated = [count for count in counts.values() if count > 1]
    return {
        'unique_values': len(counts),
        'duplicate_groups': len(repeated),
        'records_in_duplicate_groups': sum(repeated),
        'extra_duplicate_records': sum(count - 1 for count in repeated),
    }


def audit_split(path):
    ids = Counter()
    contents = Counter()
    states = Counter()
    state_labels = defaultdict(set)
    errors = Counter()
    missing = Counter()
    nulls = Counter()
    blank_text = Counter()
    labels = {field: Counter() for field in LABEL_FIELDS}
    tiers = Counter()
    amounts = []
    text_lengths = {field: [] for field in TEXT_FIELDS}
    row_count = 0
    valid_json = 0
    with path.open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                errors['blank_lines'] += 1
                continue
            row_count += 1
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                errors['invalid_json'] += 1
                continue
            valid_json += 1
            if not isinstance(row, dict):
                errors['row_not_object'] += 1
                continue
            for key in ('request_id', 'state', 'labels'):
                if key not in row:
                    missing[key] += 1
            request_id = row.get('request_id')
            if isinstance(request_id, str) and request_id:
                ids[request_id] += 1
            else:
                errors['invalid_request_id'] += 1
            contents[digest({key: value for key, value in row.items() if key != 'request_id'})] += 1
            state = row.get('state')
            truth = row.get('labels')
            if not isinstance(state, dict):
                errors['state_not_object'] += 1
            else:
                state_key = digest(state)
                states[state_key] += 1
                state_labels[state_key].add(digest(truth))
                for field in STATE_FIELDS:
                    key = f'state.{field}'
                    if field not in state:
                        missing[key] += 1
                    elif state[field] is None:
                        nulls[key] += 1
                    elif field in TEXT_FIELDS:
                        if not isinstance(state[field], str):
                            errors[f'invalid_type:{key}'] += 1
                        else:
                            text_lengths[field].append(len(state[field]))
                            if not state[field].strip():
                                blank_text[key] += 1
                    elif isinstance(state[field], bool) or not isinstance(state[field], (int, float)) or not math.isfinite(state[field]):
                        errors[f'invalid_type:{key}'] += 1
                    else:
                        amounts.append(state[field])
                tier = state.get('customer_tier')
                if isinstance(tier, str):
                    tiers[tier] += 1
            if not isinstance(truth, dict):
                errors['labels_not_object'] += 1
            else:
                for field in LABEL_FIELDS:
                    key = f'labels.{field}'
                    if field not in truth:
                        missing[key] += 1
                        continue
                    value = truth[field]
                    if value is None:
                        nulls[key] += 1
                    if field == 'route':
                        if not isinstance(value, str) or value not in QUEUES:
                            errors[f'invalid_value:{key}'] += 1
                    elif type(value) is not bool:
                        errors[f'invalid_type:{key}'] += 1
                    labels[field][str(value).lower()] += 1
    summary = {
        'file': path.name,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'bytes': path.stat().st_size,
        'records': row_count,
        'valid_json_records': valid_json,
        'schema_errors': dict(sorted(errors.items())),
        'missing_fields': dict(sorted(missing.items())),
        'null_fields': dict(sorted(nulls.items())),
        'blank_text_fields': dict(sorted(blank_text.items())),
        'labels': {field: dict(sorted(counts.items())) for field, counts in labels.items()},
        'customer_tiers': dict(sorted(tiers.items())),
        'amount': {
            'min': min(amounts) if amounts else None,
            'max': max(amounts) if amounts else None,
            'zero_count': sum(value == 0 for value in amounts),
        },
        'text_lengths': {
            field: {'min': min(values) if values else None, 'max': max(values) if values else None}
            for field, values in text_lengths.items()
        },
        'request_ids': duplicate_summary(ids),
        'content_without_request_id': duplicate_summary(contents),
        'state_only': duplicate_summary(states),
        'identical_states_with_conflicting_labels': sum(len(values) > 1 for values in state_labels.values()),
    }
    return summary, {'request_ids': ids, 'content_without_request_id': contents, 'state_only': states}


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument('--train', type=Path, default=root / 'data' / 'train.jsonl')
    parser.add_argument('--dev', type=Path, default=root / 'data' / 'dev.jsonl')
    parser.add_argument('--output', type=Path, default=root / 'evaluation' / 'data_audit.json')
    args = parser.parse_args()
    train, train_keys = audit_split(args.train)
    dev, dev_keys = audit_split(args.dev)
    overlap = {}
    for key in train_keys:
        shared = train_keys[key].keys() & dev_keys[key].keys()
        overlap[key] = {
            'shared_unique_values': len(shared),
            'train_records_matching_dev': sum(train_keys[key][value] for value in shared),
            'dev_records_matching_train': sum(dev_keys[key][value] for value in shared),
        }
    report = {'train': train, 'dev': dev, 'cross_split_overlap': overlap}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
