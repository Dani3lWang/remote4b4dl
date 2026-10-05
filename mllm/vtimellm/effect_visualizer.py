"""Plotly figures and compact HTML for the B4DL model-effect dashboard."""

from __future__ import annotations

import html
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from model_effects import (
    PAPER_REFERENCE,
    TASK_LABELS,
    TASKS,
    EvaluationSample,
    normalize_answer,
)
from training_effects import finite_number


COLORS = {
    "bg": "#ffffff",
    "panel": "#ffffff",
    "grid": "#e7e7e4",
    "text": "#242424",
    "muted": "#646464",
    "cyan": "#385a7c",
    "amber": "#92979d",
    "red": "#b54444",
    "lime": "#4f7b62",
}


def _go():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("缺少 Plotly，请安装 mllm/requirements-demo.txt") from exc
    return go


def _base_layout(figure, title: str, height: int = 360):
    figure.update_layout(
        template="plotly_white",
        autosize=True,
        title={"text": title, "x": 0.02, "xanchor": "left", "font": dict(size=17)},
        height=height,
        margin=dict(l=52, r=24, t=66, b=90),
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"], family="Times New Roman, Noto Serif CJK SC, serif", size=13),
        legend=dict(orientation="h", yanchor="top", y=-0.23, xanchor="left", x=0, font=dict(size=11)),
        hoverlabel=dict(bgcolor="#ffffff", font_color=COLORS["text"]),
    )
    figure.update_xaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"],
                        showline=True, linecolor="#777777", ticks="outside")
    figure.update_yaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"],
                        showline=True, linecolor="#777777", ticks="outside")
    return figure


def empty_effect_figure(message: str, title: str = "NO DATA"):
    go = _go()
    figure = go.Figure()
    figure.add_annotation(
        text=html.escape(message), x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(color=COLORS["muted"], size=15),
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return _base_layout(figure, title, 320)


def metric_cards_html(scores: Mapping[str, object], warnings: Sequence[str] = ()) -> str:
    metrics = (
        ("accuracy", "Accuracy"),
        ("miou", "mIoU"),
        ("bleu4", "BLEU-4"),
        ("meteor", "METEOR"),
        ("rouge_l", "ROUGE-L"),
        ("bertscore", "BERTScore"),
        ("gpt_score", "GPT score"),
    )
    cells = []
    for key, label in metrics:
        value = finite_number(scores.get(key))
        if value is None:
            formatted = "—"
            state = "is-na"
        else:
            formatted = f"{float(value):.3f}"
            state = ""
        cells.append(f'<td class="{state}">{formatted}</td>')
    warning_html = ""
    if warnings:
        warning_html = '<div class="effect-warning">' + " · ".join(
            html.escape(item) for item in warnings
        ) + "</div>"
    headers = "".join(f'<th scope="col">{html.escape(label)}</th>' for _, label in metrics)
    reference = "".join(f'<td>{PAPER_REFERENCE[key]:.3f}</td>' for key, _ in metrics)
    return (
        '<div class="metric-rail"><table class="results-table">'
        '<caption>汇总指标 · — 表示当前结果未提供该指标；论文数值为参考。</caption>'
        '<thead><tr><th scope="col">Experiment</th>' + headers + '</tr></thead>'
        '<tbody><tr class="current-run"><th scope="row">当前评测 / Current run</th>'
        + "".join(cells) + '</tr><tr class="paper-reference"><th scope="row">论文 Table 3 / Reference</th>'
        + reference + '</tr></tbody></table></div>' + warning_html
    )


def overview_metrics_figure(scores: Mapping[str, object]):
    go = _go()
    keys = ("accuracy", "miou", "bleu4", "meteor", "rouge_l", "bertscore")
    labels = ("Accuracy", "mIoU", "BLEU-4", "METEOR", "ROUGE-L", "BERTScore")
    current = [finite_number(scores.get(key)) for key in keys]
    if not any(value is not None for value in current):
        return empty_effect_figure("metrics.json 中没有 final_scores", "MODEL / PAPER")
    figure = go.Figure()
    figure.add_bar(name="当前模型", x=labels, y=current, marker_color=COLORS["cyan"])
    figure.add_bar(
        name="论文 Table 3", x=labels,
        y=[PAPER_REFERENCE[key] for key in keys], marker_color=COLORS["amber"], opacity=0.72,
    )
    figure.update_layout(barmode="group", yaxis=dict(range=[0, 1]))
    return _base_layout(figure, "(a) 汇总指标", 390)


def training_history_figure(history):
    from plotly.subplots import make_subplots
    go = _go()
    figure = make_subplots(rows=1, cols=2, subplot_titles=("训练 / 验证 Loss", "学习率"))
    for metric, name, color, column in (
        ("loss", "Training loss", COLORS["cyan"], 1),
        ("eval_loss", "Validation loss", COLORS["amber"], 1),
        ("learning_rate", "Learning rate", COLORS["lime"], 2),
    ):
        series = history.series(metric)
        if series:
            steps, values = zip(*series)
            figure.add_trace(go.Scatter(
                x=list(steps), y=list(values), name=name, mode="lines+markers",
                line=dict(color=color), marker=dict(size=4),
            ), row=1, col=column)
    for column in (1, 2):
        figure.update_xaxes(title_text="Optimizer step", row=1, col=column)
    epoch = f"{history.epoch:.2f}" if history.epoch is not None else "N/A"
    return _base_layout(figure, f"训练进度 · step {history.global_step} · epoch {epoch}", 390)


def per_task_metrics_figure(per_task_metrics: Mapping[str, object]):
    go = _go()
    figure = go.Figure()
    added = False
    metric_colors = {
        "accuracy": COLORS["cyan"],
        "miou": COLORS["amber"],
        "bleu4": "#8a6b91",
        "meteor": COLORS["lime"],
        "rouge_l": "#b58b49",
        "bertscore": COLORS["red"],
    }
    for metric, color in metric_colors.items():
        xs, ys = [], []
        for task in TASKS:
            values = per_task_metrics.get(task, {})
            value = finite_number(values.get(metric)) if isinstance(values, Mapping) else None
            if value is not None:
                xs.append(TASK_LABELS[task])
                ys.append(value)
        if xs:
            figure.add_bar(name=metric, x=xs, y=ys, marker_color=color)
            added = True
    if not added:
        return empty_effect_figure("没有 per-task 指标", "TASK METRICS")
    figure.update_layout(barmode="group", yaxis=dict(range=[0, 1]))
    return _base_layout(figure, "(b) 各任务指标", 390)


def confusion_matrix_figure(samples: Iterable[EvaluationSample], task: str):
    go = _go()
    values = [sample for sample in samples if sample.task == task]
    if not values:
        return empty_effect_figure("没有该分类任务样本", "CONFUSION MATRIX")
    labels = ("Yes", "No", "Other")

    def category(value: str) -> str:
        normalized = normalize_answer(value)
        return "Yes" if normalized == "yes" else "No" if normalized == "no" else "Other"

    matrix = np.zeros((3, 3), dtype=int)
    for sample in values:
        matrix[labels.index(category(sample.ground_truth)), labels.index(category(sample.prediction))] += 1
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[f"Pred {label}" for label in labels],
            y=[f"GT {label}" for label in labels],
            colorscale=[[0, COLORS["panel"]], [1, COLORS["cyan"]]],
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
        )
    )
    return _base_layout(figure, f"{TASK_LABELS[task]}混淆矩阵", 360)


def time_grounding_diagnostics_figure(samples: Iterable[EvaluationSample]):
    try:
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("缺少 Plotly，请安装 mllm/requirements-demo.txt") from exc
    go = _go()
    values = [sample for sample in samples if sample.task == "time_grounding"]
    if not values:
        return empty_effect_figure("没有时间定位样本", "TIME GROUNDING")
    ious = [sample.score or 0.0 for sample in values]
    pairs = [sample for sample in values if sample.gt_interval and sample.pred_interval]
    figure = make_subplots(
        rows=1, cols=2,
        subplot_titles=("单样本 IoU 分布", "GT / Prediction 开始帧"),
        horizontal_spacing=0.12,
    )
    figure.add_trace(
        go.Histogram(x=ious, nbinsx=20, marker_color=COLORS["amber"], name="IoU"), row=1, col=1,
    )
    if pairs:
        gt_starts = [sample.gt_interval[0] for sample in pairs]
        pred_starts = [sample.pred_interval[0] for sample in pairs]
        limit = max(gt_starts + pred_starts + [1])
        figure.add_trace(
            go.Scatter(
                x=gt_starts, y=pred_starts, mode="markers", name="样本",
                marker=dict(color=COLORS["cyan"], opacity=0.52, size=6),
                text=[sample.sample_id for sample in pairs],
                hovertemplate="%{text}<br>GT=%{x}, Pred=%{y}<extra></extra>",
            ), row=1, col=2,
        )
        figure.add_trace(
            go.Scatter(
                x=[0, limit], y=[0, limit], mode="lines", name="理想位置",
                line=dict(color=COLORS["red"], dash="dash"),
            ), row=1, col=2,
        )
    figure.update_xaxes(title_text="IoU", row=1, col=1)
    figure.update_xaxes(title_text="GT start", row=1, col=2)
    figure.update_yaxes(title_text="数量", row=1, col=1)
    figure.update_yaxes(title_text="Pred start", row=1, col=2)
    return _base_layout(figure, "时间定位误差诊断", 420)


def timeline_figure(
    sample: EvaluationSample,
    frame_count: int,
    current_frame: Optional[int] = None,
):
    go = _go()
    frame_count = max(1, int(frame_count))
    figure = go.Figure()
    if sample.gt_interval:
        figure.add_trace(
            go.Scatter(
                x=list(sample.gt_interval), y=[1, 1], mode="lines+markers", name="Ground Truth",
                line=dict(color=COLORS["lime"], width=18), marker=dict(size=5),
                hovertemplate="GT frame %{x}<extra></extra>",
            )
        )
    if sample.pred_interval:
        figure.add_trace(
            go.Scatter(
                x=list(sample.pred_interval), y=[0, 0], mode="lines+markers", name="Prediction",
                line=dict(color=COLORS["red"], width=18), marker=dict(size=5),
                hovertemplate="Pred frame %{x}<extra></extra>",
            )
        )
    if current_frame is not None:
        figure.add_vline(x=int(current_frame), line_color=COLORS["cyan"], line_width=2)
    figure.update_xaxes(range=[-0.5, frame_count - 0.5], dtick=max(1, frame_count // 10), title="数据集帧号（0 基）")
    figure.update_yaxes(
        tickmode="array", tickvals=[0, 1], ticktext=["PRED", "GT"], range=[-0.65, 1.65],
    )
    if not sample.gt_interval and not sample.pred_interval:
        figure.add_annotation(
            text="该样本没有可解析的帧区间", x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color=COLORS["muted"]),
        )
    return _base_layout(figure, f"时间轴 · {sample.sample_id}", 250)


def sample_diagnosis_html(sample: EvaluationSample) -> str:
    score_name = (
        "Accuracy" if sample.task in ("existence", "binary_qa")
        else "IoU" if sample.task == "time_grounding"
        else "ROUGE-L（样本诊断）"
    )
    details = [
        f"<strong>{html.escape(TASK_LABELS[sample.task])}</strong>",
        f"{score_name}: <b>{sample.score_label}</b>",
        f"状态: <b>{html.escape(sample.status)}</b>",
    ]
    if sample.scene_id:
        details.append(f"Scene: <code>{html.escape(sample.scene_id)}</code>")
    if sample.gt_interval and sample.pred_interval:
        details.extend(
            [
                f"Start offset: {sample.pred_interval[0] - sample.gt_interval[0]:+d}",
                f"End offset: {sample.pred_interval[1] - sample.gt_interval[1]:+d}",
            ]
        )
    return '<div class="diagnosis-strip">' + "".join(
        f"<span>{item}</span>" for item in details
    ) + "</div>"
