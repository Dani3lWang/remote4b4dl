"""Independent checks over the recovered baseline and the delivered output."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description='核验2026-10-28 B3类别复核产物及其基线')
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--baseline-dir', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
base = args.baseline_dir
out = args.output_dir
source = args.source
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load_lines = lambda p: [json.loads(x) for x in p.open(encoding='utf-8')]
read_csv = lambda p: list(csv.DictReader(p.open(encoding='utf-8-sig', newline='')))
summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
baseline_summary = json.loads((base / 'summary.json').read_text(encoding='utf-8'))
raw = json.loads(source.read_text(encoding='utf-8-sig'))
old = load_lines(base / 'records.jsonl')
new = load_lines(out / '逐条分析.jsonl')
assert sha(source) == summary['source_sha256'] == baseline_summary['source_sha256'] == 'e5d1fe81ede865ecf496b21c1d67185dd8aacc8fea891b598f74c363ad637d89'
assert len(old) == len(new) == summary['total'] == 16067
source_records = [(task, i, p, q, gt) for task in ('description','temporal_understanding','comprehensive_reasoning')
                  for i, (p, q, gt) in enumerate(zip(raw[task]['predictions'],raw[task]['questions'],raw[task]['ground_truths']),1)]
assert len(source_records) == len(new)
keys = set()
frame_count = 0
scope_count = 0
for a, b, expected in zip(old, new, source_records):
    task, index, text, q, gt = expected
    assert (a['task'], a['index'], a['prediction'], a['row'][-2:]) == (task,index,text,[q,gt])
    assert (b['task'], b['index'], b['prediction'], b['row'][-2:]) == (task,index,text,[q,gt])
    assert (task,index) not in keys
    keys.add((task,index))
    for mention in b['mentions'] + b['extras'] + b['uncertain']:
        start,end = mention['span']
        assert text[start:end] == mention['word']
    assert a['extras'] == b['extras'] and a['uncertain'] == b['uncertain']
    if a['mentions'] != b['mentions']:
        assert len(a['mentions']) == len(b['mentions'])
        diffs = [(x,y) for x,y in zip(a['mentions'],b['mentions']) if x != y]
        assert len(diffs) == 1
        previous, current = diffs[0]
        assert previous['category'] == current['category'] == 'pedestrian'
        assert previous['status'] == '否定存在' and current['status'] == '否定新增'
        assert {k:v for k,v in previous.items() if k not in ('status','rationale')} == {k:v for k,v in current.items() if k not in ('status','rationale')}
    if a['states'] != b['states']:
        scope_count += 1
        different = [k for k in a['states'] if a['states'][k] != b['states'][k]]
        assert different == ['pedestrian']
        assert a['states']['pedestrian'] == ['肯定提及','否定存在']
        assert b['states']['pedestrian'] == ['肯定提及','否定新增']
        assert any(m['status'] == '否定新增' and m['word'].lower() in ('pedestrian','pedestrians') for m in b['mentions'])
    if b['auto_decision']:
        frame_count += 1
        assert not b['mentions'] and not b['review_reasons'] and not any(b['states'].values())
    elif a['states'] == b['states']:
        assert a['review_reasons'] == b['review_reasons']
assert (scope_count,frame_count) == (13,147)

old_review = {(r['task'],r['index']) for r in old if r['review_reasons']}
csv_review = {(r['任务类别'],int(r['类别内序号'])) for r in read_csv(base/'review.csv')}
assert old_review == csv_review and len(old_review) == 2680
new_review = {(r['task'],r['index']) for r in new if r['review_reasons']}
pending = {(r['task'],r['index']) for r in new if r['review_reasons'] or r['auto_decision']}
assert len(new_review) == 2520 and len(pending) == 2667 and pending <= old_review
assert len(read_csv(out/'规则变化记录.csv')) == 160
assert len(read_csv(out/'需复核预测.csv')) == 2520
all_rows = read_csv(out/'全部预测_逐条类别对应.csv')
matrix = read_csv(out/'16类存在关系矩阵.csv')
assert len(all_rows) == len(matrix) == 16067
for record, row, matrix_row in zip(new, all_rows, matrix):
    assert (row['任务类别'],int(row['类别内序号']),row['预测原文']) == (record['task'],record['index'],record['prediction'])
    assert (matrix_row['任务类别'],int(matrix_row['类别内序号'])) == (record['task'],record['index'])
    for cls, values in record['states'].items():
        assert matrix_row[cls] == ('/'.join(values) or '未提及')
for task, counts in summary['tasks'].items():
    subset = [r for r in new if r['task']==task]
    assert len(subset)==counts['total']
    assert sum(bool(r['review_reasons']) for r in subset)==counts['review']
    assert sum(not r['mentions'] for r in subset)==counts['no_official_mention']
    for cls, stats in summary['classes'].items():
        for status in ('肯定提及','否定存在','否定新增','条件或可能','泛指或建议'):
            assert sum(status in r['states'][cls] for r in subset)==stats[task][status]
        assert sum(bool(r['states'][cls]) for r in subset)==stats[task]['任意提及']

by_key = {(r['task'],r['index']):r for r in new}
group_rows = read_csv(out/'人工抽查分组索引.csv')
sample_rows = read_csv(out/'人工抽查清单.csv')
seen = set()
selected = set()
group_by_digest = {}
for row in group_rows:
    task, stratum, digest = row['任务'], row['互斥主层'], row['分组SHA256']
    ids = [int(s) for s in row['任务内序号'].split('；')]
    assert digest not in group_by_digest
    group_by_digest[digest] = (task, stratum, ids)
    assert len(ids) == int(row['组内记录数'])
    for index in ids:
        key = (task,index)
        assert key in pending and key not in seen
        seen.add(key)
        r = by_key[key]
        content = (task,stratum,r['prediction'],r['row'][-2])
        expected = hashlib.sha256(json.dumps(content,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()
        assert digest == expected
    if row['抽中']=='是': selected.add(digest)
assert seen == pending
assert {r['分组SHA256'] for r in sample_rows} == selected
for row in sample_rows:
    task, stratum, ids = group_by_digest[row['分组SHA256']]
    example = by_key[(task,ids[0])]
    assert row['任务'] == task and row['互斥主层'] == stratum
    assert row['任务内序号'] == '；'.join(map(str,ids)) and int(row['本组记录数']) == len(ids)
    assert row['预测原文'] == example['prediction'] and row['原问题（仅辅助理解指代）'] == example['row'][-2]
assert len(sample_rows) == summary['review_analysis']['sample_groups'] == 101
assert sum(int(r['本组记录数']) for r in sample_rows) == summary['review_analysis']['sample_covered_records'] == 141
assert len(read_csv(out/'人工抽查分层汇总.csv')) == 35
result = {'source_sha256':sha(source), 'source_and_baseline_records':len(new),
          'old_review_csv_and_jsonl_match':len(old_review), 'mention_spans':'all_match',
          'only_expected_mention_differences':'pass', 'csv_and_category_matrix':'pass',
          'changed_new_scope':scope_count, 'changed_frame_only':frame_count,
          'new_review':len(new_review), 'sample_eligible_including_unknown_target':len(pending),
          'group_index_covers_each_candidate_once':len(seen), 'sample_groups':len(sample_rows),
          'sampled_group_record_count':141, 'class_counts_recomputed':'pass',
          'qualification':'text consistency only; no human correctness, point mask, or model-quality validation'}
(out/'独立核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
