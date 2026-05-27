#!/usr/bin/env python3
"""Compute B4DL metrics from evaluation log."""
import re
import json
import argparse


def parse_frame_range(text):
    m = re.match(r'from frame (\d+) to frame (\d+)', text.strip().lower().rstrip('.'))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def compute_binary_metrics(results):
    y_true, y_pred = [], []
    for r in results:
        gt = r['gt'].strip().lower().rstrip('.')
        pred = r['pred'].strip().lower().rstrip('.')
        if gt in ('yes', 'no') and pred in ('yes', 'no'):
            y_true.append(1 if gt == 'yes' else 0)
            y_pred.append(1 if pred == 'yes' else 0)

    n = len(y_true)
    if n == 0:
        return {"binary_count": 0}

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "binary_count": n,
        "binary_accuracy": round(correct / n * 100, 2),
        "binary_precision": round(precision * 100, 2),
        "binary_recall": round(recall * 100, 2),
        "binary_f1": round(f1 * 100, 2),
    }


def compute_frame_range_metrics(results):
    ious = []
    exact = 0
    total = 0

    for r in results:
        gt_range = parse_frame_range(r['gt'])
        pred_range = parse_frame_range(r['pred'])
        if gt_range is None:
            continue
        total += 1
        if pred_range is None:
            ious.append(0.0)
            continue

        gt_s, gt_e = gt_range
        pred_s, pred_e = pred_range
        inter = max(0, min(gt_e, pred_e) - max(gt_s, pred_s) + 1)
        union = max(gt_e, pred_e) - min(gt_s, pred_s) + 1
        iou = inter / union if union > 0 else 0.0
        ious.append(iou)

        if gt_s == pred_s and gt_e == pred_e:
            exact += 1

    if total == 0:
        return {"frame_range_count": 0}

    return {
        "frame_range_count": total,
        "frame_range_exact_match": round(exact / total * 100, 2),
        "frame_range_mean_iou": round(sum(ious) / len(ious) * 100, 2),
        "frame_range_R1@0.5": round(sum(1 for i in ious if i >= 0.5) / len(ious) * 100, 2),
        "frame_range_R1@0.7": round(sum(1 for i in ious if i >= 0.7) / len(ious) * 100, 2),
    }


def compute_categorical_metrics(results):
    y_true, y_pred = [], []
    for r in results:
        if r.get('answer_type') != 'categorical':
            continue
        gt = r['gt'].strip().lower().rstrip('.')
        pred = r['pred'].strip().lower().rstrip('.')
        y_true.append(gt)
        y_pred.append(pred)

    n = len(y_true)
    if n == 0:
        return {"categorical_count": 0}

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n

    classes = sorted(set(y_true))
    per_class_f1 = []
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1.append(f1)

    macro_f1 = sum(per_class_f1) / len(per_class_f1) if per_class_f1 else 0.0

    return {
        "categorical_count": n,
        "categorical_accuracy": round(accuracy * 100, 2),
        "categorical_macro_f1": round(macro_f1 * 100, 2),
    }


def _rouge_l(pred, ref):
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()
    m, n = len(pred_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / m if m > 0 else 0
    rec = lcs / n if n > 0 else 0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def compute_captioning_metrics(results):
    predictions = [r['pred'].strip() for r in results]
    references = [r['gt'].strip() for r in results]

    # BLEU via NLTK
    bleu_scores = {}
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        ref_lists = [[ref.split()] for ref in references]
        pred_lists = [pred.split() for pred in predictions]
        smoothie = SmoothingFunction().method4
        for n_gram in [1, 2, 3, 4]:
            weights = tuple(1.0 / n_gram for _ in range(n_gram))
            score = corpus_bleu(ref_lists, pred_lists, weights=weights, smoothing_function=smoothie)
            bleu_scores[f"BLEU-{n_gram}"] = round(score * 100, 2)
    except ImportError:
        print("[WARN] nltk not available for BLEU. Install with: pip install nltk")
        bleu_scores = {"BLEU": "nltk_not_available"}

    # ROUGE-L
    rouge_scores = [_rouge_l(p, r) for p, r in zip(predictions, references)]
    rouge_l_f1 = round(sum(rouge_scores) / len(rouge_scores) * 100, 2) if rouge_scores else 0.0

    # METEOR
    meteor = None
    try:
        from nltk.translate.meteor_score import meteor_score
        meteor_scores = [meteor_score([ref], pred) for ref, pred in zip(references, predictions)]
        meteor = round(sum(meteor_scores) / len(meteor_scores) * 100, 2)
    except ImportError:
        meteor = "nltk_not_available"
    except Exception as e:
        meteor = f"error: {e}"

    metrics = {**bleu_scores, "ROUGE-L": rouge_l_f1, "METEOR": meteor,
               "captioning_count": len(results)}
    return metrics


def print_metrics(metrics):
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--task", type=str, choices=['stage2', 'stage3'], required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = []
    with open(args.log_path) as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass

    print(f"Loaded {len(results)} results from {args.log_path}")

    all_metrics = {}

    if args.task == 'stage2':
        print("\n=== Stage2 Evaluation ===")
        binary_results = [r for r in results if r.get('answer_type') == 'binary']
        frame_results = [r for r in results if r.get('answer_type') == 'frame_range']
        cat_results = [r for r in results if r.get('answer_type') == 'categorical']

        if binary_results:
            print("\n--- Binary QA ---")
            m = compute_binary_metrics(binary_results)
            print_metrics(m)
            all_metrics.update(m)

        if frame_results:
            print("\n--- Frame Range ---")
            m = compute_frame_range_metrics(frame_results)
            print_metrics(m)
            all_metrics.update(m)

        if cat_results:
            print("\n--- Categorical ---")
            m = compute_categorical_metrics(cat_results)
            print_metrics(m)
            all_metrics.update(m)

        exact = sum(1 for r in results
                    if r['pred'].strip().lower() == r['gt'].strip().lower())
        all_metrics['overall_accuracy'] = round(exact / len(results) * 100, 2)
        print(f"\n  Overall Exact Match: {all_metrics['overall_accuracy']:.2f}%")

    elif args.task == 'stage3':
        print("\n=== Stage3 (Captioning) Evaluation ===")
        m = compute_captioning_metrics(results)
        print_metrics(m)
        all_metrics.update(m)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
