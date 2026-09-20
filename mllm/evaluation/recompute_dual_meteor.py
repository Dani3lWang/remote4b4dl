#!/usr/bin/env python
"""Recompute METEOR on saved predictions with BOTH backends, offline (no GPU).

Writes <dir>/metrics_recomputed_20260907_dual.json per results directory:
  - 'meteor'            -> NLTK-2005 (Banerjee & Lavie 2005 = paper ref [2]),
                           per-sample mean — the new reported main value
  - 'meteor_pycocoevalcap' -> Meteor-1.5 jar (COCO convention; the value the
                           B0-B3 tables reported as 'meteor')
All other metrics (accuracy/mIoU/bleu4/rouge_l/bertscore/gpt) are copied
unchanged from the existing metrics.json. Original files are untouched.

Usage (run from mllm/):
  python evaluation/recompute_dual_meteor.py [--dir DIR ...]
  # default dirs: the B0-B3 result directories
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MLLM = os.path.dirname(_HERE)
if _MLLM not in sys.path:
    sys.path.insert(0, _MLLM)

from evaluation.evaluate_model import B4DLEvaluator  # noqa: E402

DEFAULT_DIRS = [
    "eval_results/stage2_full_seqv3_mixed",        # B0
    "eval_results/stage2_full_seqv3_mixed_b1",     # B1
    "eval_results/stage2_full_seqv3_mixed_b2",     # B2
    "eval_results/stage2_full_seqv3_mixed_b3",     # B3
]
OUT_SUFFIX = "metrics_recomputed_20260907_dual.json"
COMPLEX = B4DLEvaluator.COMPLEX_TASKS
# Regression anchors (must reproduce, same code path + same predictions):
#   B0 old NLTK final (per-sample mean)        = 0.3344
#   B0 8/29 jar recompute final                = 0.1729
#   B1-B3 jar finals (their metrics.json)      = 0.1750 / 0.1747 / 0.1747
TOL = 0.0015


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", action="append", default=None,
                    help="results dir containing predictions.json + metrics.json")
    ap.add_argument("--out-suffix", default=OUT_SUFFIX)
    args = ap.parse_args()
    dirs = args.dir or DEFAULT_DIRS

    # Constructing the evaluator only resolves the bert-score registry entry
    # (no model load); METEOR paths below need no GPU.
    ev = B4DLEvaluator(meteor_backend='dual')
    failures = 0

    for d in dirs:
        pred_path = os.path.join(d, "predictions.json")
        met_path = os.path.join(d, "metrics.json")
        if not (os.path.isfile(pred_path) and os.path.isfile(met_path)):
            print(f"[skip] missing predictions.json/metrics.json in {d}")
            continue
        preds = json.load(open(pred_path))
        old = json.load(open(met_path))
        # Jar anchor: prefer the 08-29 jar recompute file when present (B0's
        # metrics.json still holds the legacy NLTK value 0.3344 in 'meteor').
        anchor_path = os.path.join(d, "metrics_recomputed_20260829.json")
        jar_anchor = json.load(open(anchor_path)) if os.path.isfile(anchor_path) else old

        per_task = json.loads(json.dumps(old.get("per_task_metrics", {})))
        finals = json.loads(json.dumps(old.get("final_scores", {})))
        jar_finals, nltk_finals = [], []
        print(f"\n=== {os.path.basename(d) or d}")
        for t in COMPLEX:
            p = preds.get(t, {}).get("predictions")
            g = preds.get(t, {}).get("ground_truths")
            if not p or not g or t not in per_task:
                print(f"  [skip] task {t}: predictions or per-task metrics missing")
                continue
            nltk_v = ev._meteor_nltk_2005(p, g)
            jar_v = ev._meteor_pycocoevalcap(p, g)
            per_task[t]["meteor"] = nltk_v              # NLTK-2005 = main
            per_task[t]["meteor_pycocoevalcap"] = jar_v
            nltk_finals.append(nltk_v)
            jar_finals.append(jar_v)
            print(f"  {t:24s} meteor={nltk_v:.4f}  meteor_pycocoevalcap={jar_v:.4f}")
        if nltk_finals:
            finals["meteor"] = sum(nltk_finals) / len(nltk_finals)
        if jar_finals:
            finals["meteor_pycocoevalcap"] = sum(jar_finals) / len(jar_finals)

        out = {"per_task_metrics": per_task, "final_scores": finals,
               "metric_backend": ev.metric_backend}
        out_path = os.path.join(d, args.out_suffix)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  -> {out_path}")
        print(f"  final meteor={finals.get('meteor'):.4f}  "
              f"meteor_pycocoevalcap={finals.get('meteor_pycocoevalcap'):.4f}")

        # Regression checks against the pre-existing values.
        old_jar = jar_anchor.get("final_scores", {}).get("meteor")
        if old_jar is not None and finals.get("meteor_pycocoevalcap") is not None:
            diff = abs(finals["meteor_pycocoevalcap"] - old_jar)
            flag = "OK" if diff <= TOL else "MISMATCH"
            if diff > TOL:
                failures += 1
            print(f"  [check jar-vs-original] {flag}: recomputed "
                  f"{finals['meteor_pycocoevalcap']:.4f} vs original "
                  f"{old_jar:.4f} (d={diff:.4f})")
        if os.path.basename(d) == "stage2_full_seqv3_mixed" \
                and finals.get("meteor") is not None:
            diff = abs(finals["meteor"] - 0.3344)
            flag = "OK" if diff <= TOL else "MISMATCH"
            if diff > TOL:
                failures += 1
            print(f"  [check nltk-vs-0.3344] {flag}: recomputed "
                  f"{finals['meteor']:.4f} vs legacy B0 NLTK 0.3344 "
                  f"(d={diff:.4f})")

    print("\n" + ("ALL CHECKS PASSED" if failures == 0
                  else f"{failures} CHECK(S) FAILED"))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
