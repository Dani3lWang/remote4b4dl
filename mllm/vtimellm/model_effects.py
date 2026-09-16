"""Lightweight loading and per-sample diagnostics for B4DL evaluations.

This module intentionally does not import the model stack or metric backends.
It is safe to use in the CPU-only Gradio viewer after inference has finished.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TASKS: Tuple[str, ...] = (
    "existence",
    "binary_qa",
    "time_grounding",
    "description",
    "temporal_understanding",
    "comprehensive_reasoning",
)
COMPLEX_TASKS = frozenset(TASKS[3:])
TASK_LABELS: Mapping[str, str] = {
    "existence": "存在性",
    "binary_qa": "二分类问答",
    "time_grounding": "时间定位",
    "description": "场景描述",
    "temporal_understanding": "时序理解",
    "comprehensive_reasoning": "综合推理",
}
TASK_ALIASES: Mapping[str, str] = {
    "binary": "binary_qa",
    "binaryqa": "binary_qa",
    "timegrounding": "time_grounding",
    "temporal": "temporal_understanding",
    "comprehensive": "comprehensive_reasoning",
}
PAPER_REFERENCE: Mapping[str, float] = {
    "accuracy": 0.762,
    "miou": 0.311,
    "bleu4": 0.095,
    "meteor": 0.275,
    "rouge_l": 0.322,
    "bertscore": 0.897,
    "gpt_score": 59.513,
}

_INTERVAL_PATTERNS = (
    re.compile(r"from\s+frame\s+(\d+)\s+to\s+frame\s+(\d+)", re.I),
    re.compile(r"frame\s+(\d+)\s+(?:to|-|and)\s+frame\s+(\d+)", re.I),
    re.compile(r"from\s+frame\s+(\d+)\s+until\s+frame\s+(\d+)", re.I),
)


def canonical_task(task: Optional[str]) -> Optional[str]:
    if task is None:
        return None
    key = task.strip().lower().replace("-", "_").replace(" ", "_")
    return TASK_ALIASES.get(key, key)


def normalize_answer(answer: Any) -> str:
    value = str(answer or "").strip().lower()
    value = re.sub(r"^(the answer is|answer:|a:)\s*", "", value)
    value = re.sub(r"[^\w\s]", "", value)
    return value.strip()


def clean_question(question: Any) -> str:
    value = str(question or "").split("<meta>", 1)[0]
    for token in ("<4DLiDAR>", "<video>"):
        value = value.replace(token, "")
    return " ".join(value.split())


def parse_frame_interval(text: Any) -> Optional[Tuple[int, int]]:
    value = str(text or "")
    for pattern in _INTERVAL_PATTERNS:
        match = pattern.search(value)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def parse_question_frames(text: Any) -> Optional[Tuple[int, int]]:
    matches = [int(value) for value in re.findall(r"frame\s+(\d+)", str(text or ""), re.I)]
    if not matches:
        return None
    return min(matches), max(matches)


def interval_iou(
    prediction: Optional[Tuple[int, int]],
    ground_truth: Optional[Tuple[int, int]],
) -> float:
    """Closed-integer-interval IoU, matching B4DLEvaluator."""
    if prediction is None or ground_truth is None:
        return 0.0
    ps, pe = prediction
    gs, ge = ground_truth
    if pe < ps or ge < gs:
        return 0.0
    intersection = min(pe, ge) - max(ps, gs) + 1
    if intersection <= 0:
        return 0.0
    union = (pe - ps + 1) + (ge - gs + 1) - intersection
    return intersection / union if union > 0 else 0.0


def _tokens(text: Any) -> List[str]:
    return re.findall(r"[\w']+", str(text or "").lower())


def rouge_l_f1(prediction: Any, ground_truth: Any) -> float:
    """Dependency-free ROUGE-L F1 used only for qualitative sample sorting."""
    pred = _tokens(prediction)
    gt = _tokens(ground_truth)
    if not pred or not gt:
        return 0.0
    previous = [0] * (len(gt) + 1)
    for pred_token in pred:
        current = [0]
        for column, gt_token in enumerate(gt, start=1):
            if pred_token == gt_token:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(pred)
    recall = lcs / len(gt)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    task: str
    source_index: int
    question: str
    ground_truth: str
    prediction: str
    scene_id: Optional[str] = None
    scene_token: Optional[str] = None
    feat_indices: Optional[Tuple[int, ...]] = None
    feat_range: Optional[Tuple[int, int]] = None
    score: Optional[float] = None
    status: str = "unscored"
    gt_interval: Optional[Tuple[int, int]] = None
    pred_interval: Optional[Tuple[int, int]] = None

    @property
    def default_frame(self) -> int:
        if self.pred_interval is not None:
            return self.pred_interval[0]
        if self.gt_interval is not None:
            return self.gt_interval[0]
        question_frames = parse_question_frames(self.question)
        return question_frames[0] if question_frames else 0

    @property
    def score_label(self) -> str:
        if self.score is None:
            return "N/A"
        if self.task in ("existence", "binary_qa"):
            return "正确" if self.score == 1.0 else "错误"
        return f"{self.score:.3f}"


def _score_sample(task: str, prediction: str, ground_truth: str):
    if task in ("existence", "binary_qa"):
        score = float(normalize_answer(prediction) == normalize_answer(ground_truth))
        return score, ("correct" if score else "error"), None, None
    if task == "time_grounding":
        pred_interval = parse_frame_interval(prediction)
        gt_interval = parse_frame_interval(ground_truth)
        score = interval_iou(pred_interval, gt_interval)
        if pred_interval is None:
            status = "unparseable"
        elif score == 1.0:
            status = "exact"
        elif score > 0.0:
            status = "overlap"
        else:
            status = "miss"
        return score, status, gt_interval, pred_interval
    score = rouge_l_f1(prediction, ground_truth)
    status = "strong" if score >= 0.5 else "partial" if score >= 0.2 else "weak"
    return score, status, None, None


def _read_json(path: Optional[str]) -> Any:
    if not path:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"文件不存在：{resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _test_item_metadata(item: Mapping[str, Any], source_index: int) -> Dict[str, Any]:
    indices = item.get("feat_indices")
    feat_range = item.get("feat_range")
    return {
        "source_index": source_index,
        "scene_id": item.get("scene_id") or item.get("id"),
        "scene_token": item.get("scene_token"),
        "feat_indices": list(indices) if indices else None,
        "feat_range": list(feat_range) if feat_range else None,
    }


def _group_test_items(test_data: Any) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    if not isinstance(test_data, list):
        return grouped
    for item in test_data:
        if not isinstance(item, Mapping):
            continue
        task = canonical_task(item.get("task"))
        conversations = item.get("conversations") or []
        if task in TASKS and len(conversations) >= 2:
            grouped[task].append(item)
    return grouped


def _legacy_metadata(
    task: str,
    questions: Sequence[str],
    ground_truths: Sequence[str],
    test_items: Sequence[Mapping[str, Any]],
) -> Tuple[List[Optional[Dict[str, Any]]], List[str]]:
    """Safely associate legacy predictions with their source QA entries."""
    warnings: List[str] = []
    lookup: Dict[Tuple[str, str], deque] = defaultdict(deque)
    for source_index, item in enumerate(test_items):
        conversations = item.get("conversations") or []
        question = clean_question(conversations[0].get("value") if conversations else "")
        gt = normalize_answer(conversations[1].get("value") if len(conversations) > 1 else "")
        lookup[(question, gt)].append((source_index, item))

    result: List[Optional[Dict[str, Any]]] = []
    unmatched = 0
    for question, ground_truth in zip(questions, ground_truths):
        key = (clean_question(question), normalize_answer(ground_truth))
        candidates = lookup.get(key)
        if not candidates:
            result.append(None)
            unmatched += 1
            continue
        source_index, item = candidates.popleft()
        result.append(_test_item_metadata(item, source_index))
    if unmatched:
        warnings.append(f"{TASK_LABELS[task]}有 {unmatched} 条旧结果无法可靠关联测试场景")
    return result, warnings


class EvaluationRepository:
    """Validated, filterable evaluation results for the visualization UI."""

    def __init__(
        self,
        samples: Iterable[EvaluationSample],
        metrics: Optional[Mapping[str, Any]] = None,
        run_metadata: Optional[Mapping[str, Any]] = None,
        warnings: Optional[Iterable[str]] = None,
    ) -> None:
        self.samples = tuple(samples)
        self.metrics = dict(metrics or {})
        self.run_metadata = dict(run_metadata or {})
        self.warnings = tuple(warnings or ())
        self._samples_by_id = {sample.sample_id: sample for sample in self.samples}
        if len(self._samples_by_id) != len(self.samples):
            raise ValueError("评测结果包含重复 sample_id")

    @classmethod
    def from_files(
        cls,
        predictions_path: Optional[str] = None,
        metrics_path: Optional[str] = None,
        test_data_path: Optional[str] = None,
    ) -> "EvaluationRepository":
        predictions = _read_json(predictions_path) if predictions_path else {}
        metrics = _read_json(metrics_path) if metrics_path else {}
        test_data = _read_json(test_data_path) if test_data_path else None
        if predictions and not isinstance(predictions, Mapping):
            raise ValueError("predictions.json 顶层必须是对象")
        if metrics and not isinstance(metrics, Mapping):
            raise ValueError("metrics.json 顶层必须是对象")

        grouped_test = _group_test_items(test_data)
        samples: List[EvaluationSample] = []
        warnings: List[str] = []
        run_metadata = predictions.get("_run", {}) if predictions else {}

        for task in TASKS:
            task_data = predictions.get(task) if predictions else None
            if not task_data:
                continue
            if not isinstance(task_data, Mapping):
                raise ValueError(f"{task} 结果必须是对象")
            preds = list(task_data.get("predictions", []))
            gts = list(task_data.get("ground_truths", []))
            questions = list(task_data.get("questions", []))
            lengths = {len(preds), len(gts), len(questions)}
            if len(lengths) != 1:
                raise ValueError(
                    f"{task} 数组长度不一致：pred={len(preds)}, gt={len(gts)}, q={len(questions)}"
                )
            metadata = task_data.get("samples")
            if metadata is not None:
                metadata = list(metadata)
                if len(metadata) != len(preds):
                    raise ValueError(
                        f"{task}.samples 长度 {len(metadata)} 与预测数 {len(preds)} 不一致"
                    )
            elif grouped_test.get(task):
                metadata, legacy_warnings = _legacy_metadata(
                    task, questions, gts, grouped_test[task]
                )
                warnings.extend(legacy_warnings)
            else:
                metadata = [None] * len(preds)
                warnings.append(f"{TASK_LABELS[task]}为旧结果格式，未提供 test_data，无法关联场景")

            for index, (prediction, ground_truth, question, raw_meta) in enumerate(
                zip(preds, gts, questions, metadata)
            ):
                raw_meta = raw_meta if isinstance(raw_meta, Mapping) else {}
                source_index = int(raw_meta.get("source_index", index))
                sample_id = str(raw_meta.get("sample_id") or f"{task}:{source_index:06d}")
                score, status, gt_interval, pred_interval = _score_sample(
                    task, str(prediction), str(ground_truth)
                )
                raw_indices = raw_meta.get("feat_indices")
                raw_range = raw_meta.get("feat_range")
                samples.append(
                    EvaluationSample(
                        sample_id=sample_id,
                        task=task,
                        source_index=source_index,
                        question=str(question),
                        ground_truth=str(ground_truth),
                        prediction=str(prediction),
                        scene_id=(str(raw_meta["scene_id"]) if raw_meta.get("scene_id") is not None else None),
                        scene_token=(str(raw_meta["scene_token"]) if raw_meta.get("scene_token") else None),
                        feat_indices=(tuple(int(v) for v in raw_indices) if raw_indices else None),
                        feat_range=(tuple(int(v) for v in raw_range) if raw_range else None),
                        score=score,
                        status=status,
                        gt_interval=gt_interval,
                        pred_interval=pred_interval,
                    )
                )
        return cls(samples, metrics, run_metadata, warnings)

    @property
    def enabled(self) -> bool:
        return bool(self.samples or self.metrics)

    @property
    def final_scores(self) -> Mapping[str, Any]:
        value = self.metrics.get("final_scores", {})
        return value if isinstance(value, Mapping) else {}

    @property
    def per_task_metrics(self) -> Mapping[str, Any]:
        value = self.metrics.get("per_task_metrics", {})
        return value if isinstance(value, Mapping) else {}

    def get(self, sample_id: str) -> EvaluationSample:
        try:
            return self._samples_by_id[str(sample_id)]
        except KeyError as exc:
            raise KeyError(f"未知评测样本：{sample_id}") from exc

    def for_task(self, task: str) -> Tuple[EvaluationSample, ...]:
        if task in (None, "", "all"):
            return self.samples
        canonical = canonical_task(task)
        return tuple(sample for sample in self.samples if sample.task == canonical)

    def filter(
        self,
        task: str = "all",
        status: str = "all",
        query: str = "",
        sort_order: str = "score_asc",
    ) -> List[EvaluationSample]:
        values = list(self.for_task(task))
        if status not in (None, "", "all"):
            values = [sample for sample in values if sample.status == status]
        needle = query.strip().lower()
        if needle:
            values = [
                sample
                for sample in values
                if needle in sample.question.lower()
                or needle in sample.ground_truth.lower()
                or needle in sample.prediction.lower()
                or needle in (sample.scene_id or "").lower()
            ]
        reverse = sort_order == "score_desc"
        if sort_order in ("score_asc", "score_desc"):
            values.sort(
                key=lambda sample: (
                    sample.score is None,
                    -(sample.score or 0.0) if reverse else (sample.score or 0.0),
                    sample.source_index,
                ),
            )
        else:
            values.sort(key=lambda sample: (TASKS.index(sample.task), sample.source_index))
        return values

    def page(
        self,
        task: str = "all",
        status: str = "all",
        query: str = "",
        sort_order: str = "score_asc",
        page: int = 0,
        page_size: int = 50,
    ) -> Tuple[List[EvaluationSample], int, int]:
        values = self.filter(task, status, query, sort_order)
        total = len(values)
        pages = max(1, math.ceil(total / page_size))
        page = max(0, min(int(page), pages - 1))
        start = page * page_size
        return values[start:start + page_size], total, page

    def task_counts(self) -> Dict[str, int]:
        return {task: len(self.for_task(task)) for task in TASKS}
