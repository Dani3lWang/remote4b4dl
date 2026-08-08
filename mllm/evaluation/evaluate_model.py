"""
B4DL Model Evaluation - metrics library
=======================================
Implements the evaluation metrics described in the B4DL paper (Section 3.1 /
5.1). Usable both as a library (import B4DLEvaluator) and as a CLI:

    from evaluation.evaluate_model import B4DLEvaluator
    ev = B4DLEvaluator()
    metrics = ev.evaluate_all(results)

Metrics (paper Table 3):
  Simple Tasks
    - Existence / Binary QA : Accuracy (exact answer matching)
    - Time Grounding        : mIoU (overlap of predicted vs GT temporal segments)
  Complex Tasks (Description / Temporal / Comprehensive Reasoning)
    - BLEU-4, METEOR, ROUGE-L, BERTScore, (optional) GPT-4o Score (0-100)

Final aggregation (paper Section 5.1):
  Each metric is computed per task, then averaged across tasks to obtain the
  final score. Accuracy = mean(Existence, Binary QA). mIoU = Time Grounding.
  The remaining 5 metrics are averaged over the 3 complex tasks.

CLI usage:
    python evaluate_model.py --predictions predictions.json --ground_truth ground_truth.json
    python evaluate_model.py --demo
"""

import os
import re
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# --------------------------------------------------------------------------
# Optional metric dependencies. Each is guarded so the evaluator still works
# (returning 0.0 for missing metrics) if a library is absent.
# --------------------------------------------------------------------------
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError:  # older NLTK versions
        from nltk.translate.meteor_score import single_meteor_score as _sms

        def meteor_score(refs, hyp):
            return _sms(refs[0], hyp)
    HAS_NLTK = True

    def _ensure_nltk_data():
        """Download required NLTK data if missing (idempotent, quiet).

        Only downloads when the resource is genuinely absent (LookupError).
        Corrupted files are noted but NOT auto-repaired to avoid network hangs;
        the user should manually run: python -c \"import nltk; nltk.download('punkt')\"
        """
        _required = {
            'tokenizers/punkt_tab': 'punkt_tab',
            'tokenizers/punkt': 'punkt',
            'corpora/wordnet': 'wordnet',
            'corpora/omw-1.4': 'omw-1.4',
        }
        for resource_path, download_name in _required.items():
            try:
                nltk.data.find(resource_path)
            except LookupError:
                # Resource genuinely not found — attempt download (may hang offline)
                try:
                    nltk.download(download_name, quiet=True)
                except Exception:
                    pass
            except Exception:
                # Corrupted resource (e.g. BadZipFile) — warn, don't auto-download
                pass
    _ensure_nltk_data()

except ImportError:
    HAS_NLTK = False
    print("Warning: NLTK not installed. BLEU and METEOR scores unavailable.")

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("Warning: rouge-score not installed. ROUGE-L score unavailable.")

try:
    from bert_score import score as bert_score_fn
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("Warning: bert-score not installed. BERTScore unavailable.")

try:
    from openai import OpenAI          # openai>=1.0 client API
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai not installed. GPT score unavailable.")


# --------------------------------------------------------------------------
# Paper Table 9 prompt for the reference-free GPT-4o evaluator.
# --------------------------------------------------------------------------
GPT_EVAL_PROMPT = """You are an expert evaluator for semantic answer quality. You will be given a set of question-answer-ground truth (Q/A/GT) triplets. Your task is to evaluate how semantically close the given answer (A) is to the ground truth (GT), and how well it responds to the question (Q).
Focus only on:
- Whether the answer conveys the same or similar meaning as the ground truth.
- Whether the answer correctly and sufficiently addresses the question.
- Do not consider wording, phrasing, or grammar unless they change the meaning. Score the answer from 0 to 100:
- 100: Fully aligned with GT in meaning, complete and relevant.
- 80-99: Mostly aligned with minor omissions or slight deviation.
- 60-79: Partially aligned; captures key ideas but misses or misrepresents some.
- 40-59: Limited relevance or meaning overlap with GT.
- 20-39: Barely related to the GT.
- 0-19: Meaningless or unrelated.
EXAMPLES
Question: What changes occur to the car in front over the frames?
Answer: The car in front moves slightly forward from frame 000 to frame 008.
GT: The car in front moves slightly forward until it stops again from frame 0 to frame 8.
Score: 90
Question: From frame 004 to frame 006, how does the building structure in the back view affect the scene?
Answer: The building structure on the left side of the road in the back view remains constant, providing a stable reference point for the ego vehicle's navigation.
GT: The building structure remains fixed, providing a static backdrop in the scene without impacting the dynamics.
Score: 100
INSTRUCTIONS
Question: {q} GT: {ref} Answer: {pred}
Please provide a score between 0 and 100 based on the quality of the prediction compared to the reference.
The score should reflect how well the prediction aligns with the reference in terms of semantic similarity and relevance to the question. Only provide the score without any additional text."""


class B4DLEvaluator:
    """Implements B4DL evaluation metrics."""

    SIMPLE_TASKS = ['existence', 'binary_qa', 'time_grounding']
    COMPLEX_TASKS = ['description', 'temporal_understanding', 'comprehensive_reasoning']
    ALL_TASKS = SIMPLE_TASKS + COMPLEX_TASKS

    # Friendly display name -> internal task key
    TASK_ALIASES = {
        ' existence': 'existence',
        'binary': 'binary_qa', 'binary_qa': 'binary_qa', 'binaryqa': 'binary_qa',
        'time_grounding': 'time_grounding', 'timegrounding': 'time_grounding',
        'description': 'description',
        'temporal': 'temporal_understanding',
        'temporal_understanding': 'temporal_understanding',
        'comprehensive': 'comprehensive_reasoning',
        'comprehensive_reasoning': 'comprehensive_reasoning',
    }

    def __init__(self, use_gpt: bool = False, gpt_api_key: Optional[str] = None,
                 gpt_model: str = "gpt-4o"):
        self.use_gpt = use_gpt and HAS_OPENAI
        self.gpt_model = gpt_model
        self._client = OpenAI(api_key=gpt_api_key) if (self.use_gpt and gpt_api_key) else None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) if HAS_ROUGE else None

    # ------------------------------------------------------------------ utils
    @classmethod
    def canonical_task(cls, task: str) -> str:
        """Map any common spelling to one of the 6 canonical task keys."""
        if task is None:
            return task
        key = task.strip().lower().replace('-', '_').replace(' ', '_')
        return cls.TASK_ALIASES.get(key, key)

    @staticmethod
    def normalize_answer(answer: str) -> str:
        """Normalize for exact-match accuracy (Existence / Binary QA)."""
        answer = answer.strip().lower()
        answer = re.sub(r'^(the answer is|answer:|a:)\s*', '', answer)
        answer = re.sub(r'[^\w\s]', '', answer)
        return answer.strip()

    @staticmethod
    def extract_time_grounding_frames(text: str) -> Dict[str, int]:
        """Extract {start_frame, end_frame} from text like 'from frame 006 to frame 014'."""
        for pat in (r'from\s+frame\s+(\d+)\s+to\s+frame\s+(\d+)',
                    r'frame\s+(\d+)\s+(?:to|-|and)\s+frame\s+(\d+)',
                    r'from\s+frame\s+(\d+)\s+until\s+frame\s+(\d+)'):
            m = re.search(pat, text.lower())
            if m:
                return {'start_frame': int(m.group(1)),
                        'end_frame': int(m.group(2))}
        return {'start_frame': 0, 'end_frame': 0}

    # --------------------------------------------------------- simple metrics
    def compute_accuracy(self, predictions: List[str], ground_truths: List[str]) -> float:
        if not predictions:
            return 0.0
        correct = sum(1 for p, g in zip(predictions, ground_truths)
                      if self.normalize_answer(p) == self.normalize_answer(g))
        return correct / len(predictions)

    def compute_miou(self, predictions: List[Dict], ground_truths: List[Dict]) -> float:
        """Mean IoU over [start_frame, end_frame] temporal segments."""
        ious = []
        for pred, gt in zip(predictions, ground_truths):
            ps, pe = pred.get('start_frame', 0), pred.get('end_frame', 0)
            gs, ge = gt.get('start_frame', 0), gt.get('end_frame', 0)
            inter = max(0, min(pe, ge) - max(ps, gs))
            union = (pe - ps) + (ge - gs) - inter
            ious.append(inter / union if union > 0 else 0.0)
        return float(np.mean(ious)) if ious else 0.0

    # --------------------------------------------------------- complex metrics
    def compute_bleu4(self, predictions: List[str], ground_truths: List[str]) -> float:
        if not HAS_NLTK or not predictions:
            return 0.0
        smooth = SmoothingFunction().method1
        scores = []
        for pred, gt in zip(predictions, ground_truths):
            pred_tokens = word_tokenize(pred.lower())
            gt_tokens = [word_tokenize(gt.lower())]
            scores.append(sentence_bleu(gt_tokens, pred_tokens,
                                        weights=(0.25, 0.25, 0.25, 0.25),
                                        smoothing_function=smooth))
        return float(np.mean(scores)) if scores else 0.0

    def compute_meteor(self, predictions: List[str], ground_truths: List[str]) -> float:
        if not HAS_NLTK or not predictions:
            return 0.0
        scores = []
        for pred, gt in zip(predictions, ground_truths):
            pred_tokens = word_tokenize(pred.lower())
            gt_tokens = word_tokenize(gt.lower())
            scores.append(meteor_score([gt_tokens], pred_tokens))
        return float(np.mean(scores)) if scores else 0.0

    def compute_rouge_l(self, predictions: List[str], ground_truths: List[str]) -> float:
        if not HAS_ROUGE or not self.rouge_scorer or not predictions:
            return 0.0
        scores = [self.rouge_scorer.score(gt, pred)['rougeL'].fmeasure
                  for pred, gt in zip(predictions, ground_truths)]
        return float(np.mean(scores)) if scores else 0.0

    def compute_bertscore(self, predictions: List[str], ground_truths: List[str]) -> float:
        if not HAS_BERTSCORE or not predictions:
            return 0.0
        try:
            _, _, F1 = bert_score_fn(predictions, ground_truths, lang='en', verbose=False)
            return float(F1.mean().item())
        except Exception as e:
            print(f"BERTScore computation failed: {e}")
            return 0.0

    def compute_gpt_score(self, predictions: List[str], ground_truths: List[str],
                          questions: List[str], batch_size: int = 16,
                          max_retries: int = 3) -> float:
        """Reference-free GPT-4o scoring on a 0..100 scale (paper Table 9)."""
        if not self.use_gpt or not self._client or not predictions:
            return 0.0
        scores = []
        for pred, gt, q in zip(predictions, ground_truths, questions):
            prompt = GPT_EVAL_PROMPT.format(q=q, ref=gt, pred=pred)
            score = None
            for _ in range(max_retries):
                try:
                    resp = self._client.chat.completions.create(
                        model=self.gpt_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=10, temperature=0,
                    )
                    raw = resp.choices[0].message.content.strip()
                    m = re.search(r'\d+', raw)
                    if m:
                        score = min(100, max(0, int(m.group())))
                        break
                except Exception as e:
                    print(f"GPT scoring failed for sample: {e}")
                    break
            scores.append(score if score is not None else 0.0)
        return float(np.mean(scores)) if scores else 0.0

    # ----------------------------------------------------- per-task evaluate
    def evaluate_task(self, task_name: str, predictions: List[Any],
                      ground_truths: List[Any],
                      questions: Optional[List[str]] = None) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        task = self.canonical_task(task_name)

        if task in ('existence', 'binary_qa'):
            metrics['accuracy'] = self.compute_accuracy(predictions, ground_truths)

        elif task == 'time_grounding':
            pred_frames = [self.extract_time_grounding_frames(p) for p in predictions]
            gt_frames = [self.extract_time_grounding_frames(gt) for gt in ground_truths]
            metrics['miou'] = self.compute_miou(pred_frames, gt_frames)

        elif task in self.COMPLEX_TASKS:
            metrics['bleu4'] = self.compute_bleu4(predictions, ground_truths)
            metrics['meteor'] = self.compute_meteor(predictions, ground_truths)
            metrics['rouge_l'] = self.compute_rouge_l(predictions, ground_truths)
            metrics['bertscore'] = self.compute_bertscore(predictions, ground_truths)
            if questions:
                metrics['gpt_score'] = self.compute_gpt_score(predictions, ground_truths, questions)
        else:
            raise ValueError(f"Unknown task: {task_name}")
        return metrics

    def evaluate_all(self, results: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
        """results: {canonical_task: {'predictions':[...], 'ground_truths':[...],
        'questions':[...]}} (optionally any alias key)."""
        all_metrics: Dict[str, Dict[str, float]] = {}
        for task_name in self.ALL_TASKS:
            task_data = results.get(task_name)
            if not task_data:
                continue
            preds = task_data.get('predictions', [])
            gts = task_data.get('ground_truths', [])
            qs = task_data.get('questions', [])
            if preds and gts:
                m = self.evaluate_task(task_name, preds, gts, qs)
                all_metrics[task_name] = m
                print(f"\n{task_name}:")
                for k, v in m.items():
                    print(f"  {k}: {v:.4f}")
        return all_metrics

    # --------------------------------------------------- paper aggregation
    def compute_final_score(self, all_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Final aggregated scores (paper Section 5.1)."""
        final: Dict[str, float] = {}

        accs = []
        for t in ('existence', 'binary_qa'):
            if t in all_metrics and 'accuracy' in all_metrics[t]:
                accs.append(all_metrics[t]['accuracy'])
        if accs:
            final['accuracy'] = float(np.mean(accs))

        if 'time_grounding' in all_metrics and 'miou' in all_metrics['time_grounding']:
            final['miou'] = all_metrics['time_grounding']['miou']

        # Complex tasks: average each metric across the 3 complex tasks.
        per_metric = defaultdict(list)
        for t in self.COMPLEX_TASKS:
            if t in all_metrics:
                for k, v in all_metrics[t].items():
                    per_metric[k].append(v)
        for k, vals in per_metric.items():
            final[k] = float(np.mean(vals))

        return final

    @staticmethod
    def save_results(all_metrics, final_scores, output_path):
        with open(output_path, 'w') as f:
            json.dump({'per_task_metrics': all_metrics,
                       'final_scores': final_scores}, f, indent=2)
        print(f"Results saved to {output_path}")


# --------------------------------------------------------------------------
# I/O helpers for the CLI / for test_b4dl.py
# --------------------------------------------------------------------------
def load_predictions(pred_path: str) -> Dict[str, Dict]:
    """Load predictions JSON in the format written by test_b4dl.py:
       {canonical_task: {'predictions':[...], 'ground_truths':[...],
                          'questions':[...]}}"""
    with open(pred_path, 'r') as f:
        return json.load(f)


def load_ground_truth(gt_path: str) -> Dict[str, Dict]:
    with open(gt_path, 'r') as f:
        return json.load(f)


def merge_pred_gt(predictions: Dict, ground_truth: Dict, evaluator: B4DLEvaluator) -> Dict:
    """Combine a predictions file and a ground-truth file into the
    evaluate_all() input shape. Either file may carry subsets of the 6 tasks."""
    merged = {}
    keys = set(predictions.keys()) | set(ground_truth.keys())
    for k in keys:
        ck = evaluator.canonical_task(k)
        p = predictions.get(k, {})
        g = ground_truth.get(k, {})
        merged[ck] = {
            'predictions': p.get('predictions', []),
            'ground_truths': g.get('ground_truths', p.get('ground_truths', [])),
            'questions': p.get('questions', g.get('questions', [])),
        }
    return merged


def create_sample_data():
    return {
        'existence': {
            'predictions': ['CAR', 'MOTORCYCLE', 'PEDESTRIAN'],
            'ground_truths': ['CAR', 'MOTORCYCLE', 'PEDESTRIAN'],
            'questions': ['Which object was behind the ego vehicle?',
                          'What vehicle was in front?',
                          'Who was crossing the street?']},
        'binary_qa': {
            'predictions': ['Yes', 'No', 'Yes'],
            'ground_truths': ['Yes', 'No', 'Yes'],
            'questions': ['Was there a motorcycle in front?',
                          'Did any bicycle appear?',
                          'Were pedestrians visible?']},
        'time_grounding': {
            'predictions': ['from frame 006 to frame 014',
                            'from frame 020 to frame 030',
                            'from frame 000 to frame 008'],
            'ground_truths': ['from frame 006 to frame 014',
                              'from frame 020 to frame 030',
                              'from frame 000 to frame 008'],
            'questions': ['When is the overpass visible?',
                          'When did the pedestrian cross?',
                          'When was the car moving?']},
        'description': {
            'predictions': ['The scene shows an urban street with vehicles and pedestrians.',
                            'The LiDAR captures a construction zone with barriers.',
                            'The ego vehicle is moving through an intersection.'],
            'ground_truths': ['The scene shows an urban street with vehicles and pedestrians moving.',
                              'The LiDAR captures a construction zone with barriers and traffic cones.',
                              'The ego vehicle is moving through an intersection with traffic lights.'],
            'questions': ['Describe the LiDAR sequence.',
                          'What does the scene show?',
                          "Describe the ego vehicle's path."]},
        'temporal_understanding': {
            'predictions': ['The car moves forward over time.',
                            'The pedestrian walks from left to right.',
                            'The bus remains stationary.'],
            'ground_truths': ['The car moves forward and slightly to the left.',
                              'The pedestrian walks from left to right crossing the street.',
                              'The bus remains stationary at the bus stop.'],
            'questions': ['How does the car move?',
                          'What is the pedestrian doing?',
                          'What happens to the bus?']},
        'comprehensive_reasoning': {
            'predictions': ['The driver should be cautious of pedestrians crossing.',
                            'The construction zone requires careful navigation.',
                            'Traffic flow is normal with some congestion ahead.'],
            'ground_truths': ['The driver should be cautious of pedestrians crossing from the left.',
                              'The construction zone requires careful navigation around barriers.',
                              'Traffic flow is normal with some congestion ahead due to construction.'],
            'questions': ['What should the driver be aware of?',
                          'How should the driver navigate?',
                          'Describe the traffic conditions.']},
    }


def main():
    parser = argparse.ArgumentParser(description='B4DL Model Evaluation')
    parser.add_argument('--predictions', type=str, help='Path to predictions JSON file')
    parser.add_argument('--ground_truth', type=str, help='Path to ground truth JSON file')
    parser.add_argument('--output', type=str, default='evaluation_results.json')
    parser.add_argument('--use_gpt', action='store_true', help='Use GPT-4o for scoring')
    parser.add_argument('--gpt_api_key', type=str, default=os.environ.get('OPENAI_API_KEY'))
    parser.add_argument('--gpt_model', type=str, default='gpt-4o')
    parser.add_argument('--demo', action='store_true', help='Run with sample data')
    args = parser.parse_args()

    evaluator = B4DLEvaluator(use_gpt=args.use_gpt, gpt_api_key=args.gpt_api_key,
                              gpt_model=args.gpt_model)

    print("=" * 50)
    print("B4DL Model Evaluation")
    print("=" * 50)

    if args.demo:
        print("Running evaluation with sample data...")
        results = create_sample_data()
    else:
        if not args.predictions or not args.ground_truth:
            print("Error: provide --predictions and --ground_truth, or use --demo")
            return
        results = merge_pred_gt(load_predictions(args.predictions),
                                load_ground_truth(args.ground_truth), evaluator)

    all_metrics = evaluator.evaluate_all(results)
    final_scores = evaluator.compute_final_score(all_metrics)

    print("\n" + "=" * 50)
    print("Final Scores")
    print("=" * 50)
    for k, v in final_scores.items():
        print(f"{k}: {v:.4f}")

    evaluator.save_results(all_metrics, final_scores, args.output)

    # Paper Table 3 reference
    print("\n" + "=" * 50)
    print("Paper reference (B4DL, Table 3)")
    print("=" * 50)
    paper = {'accuracy': 0.762, 'miou': 0.311, 'bleu4': 0.095,
             'rouge_l': 0.322, 'meteor': 0.275, 'bertscore': 0.897,
             'gpt_score': 59.513}
    for k, v in paper.items():
        got = final_scores.get(k)
        tag = f"{got:.4f}" if got is not None else "n/a"
        print(f"  {k:12s}  paper={v:<8}  ours={tag}")


if __name__ == '__main__':
    main()