import json, re
from collections import Counter

with open('/root/autodl-tmp/wql/mmb4dl/mllm/evaluation/predictions_full.json') as f:
    data = json.load(f)

print("=" * 60)
print("1. 答案模式分布")
print("=" * 60)

for task in ['existence', 'binary_qa', 'time_grounding', 'description', 'temporal_understanding', 'comprehensive_reasoning']:
    info = data[task]
    preds = info['predictions']
    gts = info['ground_truths']

    yes_count = sum(1 for p in preds if str(p).strip() == 'Yes.')
    no_count = sum(1 for p in preds if str(p).strip() == 'No.')
    time_count = sum(1 for p in preds if 'from frame' in str(p))
    short_obj = sum(1 for p in preds if len(str(p)) <= 20 and str(p).endswith('.') and 'from' not in str(p))
    long_desc = sum(1 for p in preds if len(str(p)) > 50)

    print(f"\n{task} ({len(preds)} 条):")
    print(f"  'Yes.': {yes_count} ({100*yes_count/len(preds):.1f}%)")
    print(f"  'No.':  {no_count} ({100*no_count/len(preds):.1f}%)")
    print(f"  time_pattern: {time_count} ({100*time_count/len(preds):.1f}%)")
    print(f"  短物体名: {short_obj} ({100*short_obj/len(preds):.1f}%)")
    print(f"  长描述:   {long_desc} ({100*long_desc/len(preds):.1f}%)")

print()
print("=" * 60)
print("2. existence 任务错误分析")
print("=" * 60)

info = data['existence']
preds = info['predictions']
gts = info['ground_truths']

gt_yes_count = sum(1 for g in gts if str(g).strip() == 'Yes.')
gt_no_count = sum(1 for g in gts if str(g).strip() == 'No.')
pred_yes_count = sum(1 for p in preds if str(p).strip() == 'Yes.')
pred_no_count = sum(1 for p in preds if str(p).strip() == 'No.')

# Confusion matrix
tp_yes = sum(1 for p,g in zip(preds,gts) if str(p).strip()=='Yes.' and str(g).strip()=='Yes.')
tp_no = sum(1 for p,g in zip(preds,gts) if str(p).strip()=='No.' and str(g).strip()=='No.')
fp_yes = sum(1 for p,g in zip(preds,gts) if str(p).strip()=='Yes.' and str(g).strip()!='Yes.')
fp_no = sum(1 for p,g in zip(preds,gts) if str(p).strip()=='No.' and str(g).strip()!='No.')

print(f"GT 分布: Yes={gt_yes_count} ({100*gt_yes_count/len(gts):.1f}%), No={gt_no_count}")
print(f"Pred 分布: Yes={pred_yes_count} ({100*pred_yes_count/len(preds):.1f}%), No={pred_no_count}")
print(f"Yes 精确率: {tp_yes}/{pred_yes_count} = {tp_yes/pred_yes_count:.3f}" if pred_yes_count else "N/A")
print(f"No 精确率:  {tp_no}/{pred_no_count} = {tp_no/pred_no_count:.3f}" if pred_no_count else "N/A")
print(f"Yes 召回率: {tp_yes}/{gt_yes_count} = {tp_yes/gt_yes_count:.3f}" if gt_yes_count else "N/A")
print(f"No 召回率:  {tp_no}/{gt_no_count} = {tp_no/gt_no_count:.3f}" if gt_no_count else "N/A")

# Non-Yes/No answers in existence
other_preds = [(p,g) for p,g in zip(preds,gts) if str(p).strip() not in ('Yes.','No.')]
if other_preds:
    print(f"\n非 Yes/No 预测 ({len(other_preds)} 条):")
    for p, g in other_preds[:10]:
        print(f"  pred={repr(str(p)[:60])}")
        print(f"  gt=  {repr(str(g)[:60])}")

print()
print("=" * 60)
print("3. binary_qa 偏差分析")
print("=" * 60)

info = data['binary_qa']
preds = info['predictions']
gts = info['ground_truths']

pred_yes = sum(1 for p in preds if str(p).strip()=='Yes.')
pred_no = sum(1 for p in preds if str(p).strip()=='No.')
gt_yes = sum(1 for g in gts if str(g).strip()=='Yes.')
gt_no = sum(1 for g in gts if str(g).strip()=='No.')

print(f"GT:  Yes={gt_yes} ({100*gt_yes/len(gts):.1f}%), No={gt_no}")
print(f"Pred: Yes={pred_yes} ({100*pred_yes/len(preds):.1f}%), No={pred_no}")

# Accuracy by ground truth
yes_acc = sum(1 for p,g in zip(preds,gts) if str(p).strip()==str(g).strip() and str(g).strip()=='Yes.')
no_acc = sum(1 for p,g in zip(preds,gts) if str(p).strip()==str(g).strip() and str(g).strip()=='No.')
print(f"Yes 准确率: {yes_acc}/{gt_yes} = {yes_acc/gt_yes:.3f}" if gt_yes else "N/A")
print(f"No 准确率:  {no_acc}/{gt_no} = {no_acc/gt_no:.3f}" if gt_no else "N/A")

print()
print("=" * 60)
print("4. time_grounding 帧范围分析")
print("=" * 60)

info = data['time_grounding']
preds = info['predictions']
gts = info['ground_truths']

pred_frames = Counter()
gt_frames = Counter()
for p_str in preds:
    m = re.search(r'from frame (\d+) to frame (\d+)', str(p_str))
    if m:
        pred_frames[(int(m.group(1)), int(m.group(2)))] += 1
for g_str in gts:
    m = re.search(r'from frame (\d+) to frame (\d+)', str(g_str))
    if m:
        gt_frames[(int(m.group(1)), int(m.group(2)))] += 1

print(f"不同帧范围数: Pred={len(pred_frames)}, GT={len(gt_frames)}")
print(f"Top 5 预测帧范围:")
for (s,e), c in pred_frames.most_common(5):
    print(f"  [{s:03d}-{e:03d}]: {c} ({100*c/len(preds):.1f}%)")
print(f"Top 5 GT 帧范围:")
for (s,e), c in gt_frames.most_common(5):
    print(f"  [{s:03d}-{e:03d}]: {c} ({100*c/len(gts):.1f}%)")

print()
print("=" * 60)
print("5. 复杂任务输出质量分析")
print("=" * 60)

for task in ['description', 'temporal_understanding', 'comprehensive_reasoning']:
    info = data[task]
    preds = info['predictions']

    lengths = [len(str(p)) for p in preds]
    avg_len = sum(lengths)/len(lengths)

    # Count unique first sentences (to detect template overuse)
    first_sents = Counter()
    for p in preds:
        s = str(p).split('.')[0][:80] if '.' in str(p) else str(p)[:80]
        first_sents[s] += 1

    top_template = first_sents.most_common(1)[0]

    print(f"\n{task}:")
    print(f"  平均长度: {avg_len:.0f} chars")
    print(f"  最常见开头: '{top_template[0]}...' ({top_template[1]}次, {100*top_template[1]/len(preds):.1f}%)")
    print(f"  不同开头数: {len(first_sents)}")

print()
print("=" * 60)
print("6. 各类任务典型样例")
print("=" * 60)

for task in ['existence', 'binary_qa', 'time_grounding', 'description', 'temporal_understanding', 'comprehensive_reasoning']:
    info = data[task]
    preds = info['predictions']
    gts = info['ground_truths']

    print(f"\n--- {task} ---")
    # Find GOOD predictions (len > 30 for complex tasks, exact match for simple)
    if task in ('existence', 'binary_qa'):
        matches = [(i,p,g) for i,(p,g) in enumerate(zip(preds,gts)) if str(p).strip()==str(g).strip()]
        samples = matches[:2] + matches[len(matches)//2:len(matches)//2+1]
    else:
        samples = [(i,preds[i],gts[i]) for i in [0, len(preds)//3, 2*len(preds)//3]]

    for i, p, g in samples[:3]:
        print(f"  [{i}] pred: {repr(str(p)[:100])}")
        print(f"       gt:   {repr(str(g)[:100])}")
