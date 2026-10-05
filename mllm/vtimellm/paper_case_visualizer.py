"""Publication-ready qualitative case figures for B4DL.

The renderer is deliberately independent from Gradio and the model stack.  It
loads already indexed nuScenes frames, projects available 3D annotations into
front/back cameras, draws a static LiDAR BEV, and composes a paper-style figure
that can be exported as PNG or PDF.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from .paper_scene_visualizer import TARGET_PURPLE, render_lidar_scene
except ImportError:
    from paper_scene_visualizer import TARGET_PURPLE, render_lidar_scene

try:  # package import
    from .lidar_visualizer import (
        BOX_EDGES,
        CATEGORY_COLORS,
        FrameData,
        NuScenesSceneRepository,
        category_group,
    )
except ImportError:  # direct script / demo_gradio import
    from lidar_visualizer import (  # type: ignore
        BOX_EDGES,
        CATEGORY_COLORS,
        FrameData,
        NuScenesSceneRepository,
        category_group,
    )


PAPER_BG = "#ffffff"
PAPER_PANEL = "#ffffff"
PAPER_INK = "#172028"
PAPER_MUTED = "#66727c"
PAPER_LINE = "#9da6ac"
PAPER_YELLOW = "#ffe84a"
PAPER_GREEN = "#70f08b"
PAPER_ERROR = "#c51e28"
EVIDENCE_TOP = 110
EVIDENCE_CELL_H = 196
EVIDENCE_GAP = 10


@dataclass(frozen=True)
class AnswerPanel:
    label: str
    answer: str
    error_phrases: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PaperCaseArtifact:
    image: Image.Image
    png_path: str
    pdf_path: str
    frame_indices: Tuple[int, ...]


@dataclass(frozen=True)
class TemporalCase:
    question: str
    answers: Tuple[AnswerPanel, ...]
    ground_truth: str = ""


@dataclass(frozen=True)
class ReasoningCase:
    task_label: str
    scene_token: str
    frame_index: int
    question: str
    answer: str
    target_instances: Tuple[str, ...]
    answer_label: str = "B4DL model (Ours)"
    ground_truth: str = ""
    sample_id: str = ""


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        ("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        ("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc" if bold else
         "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def select_frame_indices(
    frame_count: int,
    count: int = 5,
    requested: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    """Validate explicit frame indices or choose evenly spaced scene frames."""
    if frame_count < 2:
        raise ValueError("论文案例图至少需要 2 个可用场景帧")
    if requested is not None and len(requested) > 0:
        indices = tuple(int(value) for value in requested)
        if len(indices) < 2 or len(indices) > 8:
            raise ValueError("论文案例图需要 2 到 8 个帧号")
        if len(set(indices)) != len(indices):
            raise ValueError("帧号不能重复")
        invalid = [value for value in indices if value < 0 or value >= frame_count]
        if invalid:
            raise ValueError(
                f"帧号 {invalid} 超出场景范围 [0, {frame_count - 1}]"
            )
        return tuple(sorted(indices))
    count = max(2, min(int(count), frame_count, 8))
    values = np.linspace(0, frame_count - 1, num=count)
    return tuple(int(round(value)) for value in values)


def parse_frame_indices(value: str, frame_count: int, count: int = 5) -> Tuple[int, ...]:
    """Parse ``0, 8, 16, 24, 32`` or fall back to even spacing when blank."""
    text = str(value or "").strip()
    if not text:
        return select_frame_indices(frame_count, count=count)
    parts = [part.strip() for part in re.split(r"[,，;；\s]+", text) if part.strip()]
    try:
        requested = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("帧号必须是用逗号分隔的整数") from exc
    return select_frame_indices(frame_count, count=count, requested=requested)


def _fit_cell(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    # Preserve the full field of view: cropping could remove the evidence target.
    return ImageOps.pad(image.convert("RGB"), size, method=Image.Resampling.LANCZOS,
                        color="#eef0f0")


def scene_target_choices(repository: NuScenesSceneRepository, scene_token: str):
    """List stable object instances, rather than frame-local annotation tokens."""
    if not scene_token:
        return []
    scene = repository.get_scene(scene_token)
    instances = {}
    for sample_token in scene.sample_tokens:
        sample = repository.nusc.get("sample", sample_token)
        for token in sample.get("anns", ()):
            annotation = repository.nusc.get("sample_annotation", token)
            instance = annotation.get("instance_token")
            if instance:
                category = annotation.get("category_name", "object")
                instances[instance] = f"{category} · {instance[:12]}…"
    return sorted((label, token) for token, label in instances.items())


def _frame_target_colors(repository, frame, instances: Mapping[str, str]) -> Dict[str, str]:
    colors = {}
    sample = repository.nusc.get("sample", frame.sample_token)
    tokens = set(sample.get("anns", ())) | {box.token for box in frame.boxes}
    for token in tokens:
        try:
            annotation = repository.nusc.get("sample_annotation", token)
        except KeyError:
            continue
        color = instances.get(annotation.get("instance_token"))
        if color:
            colors[token] = color
    return colors


def render_bev_image(
    frame: FrameData,
    size: Tuple[int, int] = (640, 320),
    range_m: float = 52.0,
    show_boxes: bool = True,
    show_tracks: bool = True,
    target_colors: Optional[Mapping[str, str]] = None,
) -> Image.Image:
    """Render a compact, dependency-free top-down LiDAR panel."""
    width, height = size
    image = Image.new("RGB", size, "#e8ecec")
    draw = ImageDraw.Draw(image)
    cx, cy = width / 2.0, height / 2.0
    scale = min(width, height) / (2.0 * float(range_m))

    for radius in (10, 20, 30, 40, 50):
        r = radius * scale
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline="#c8d0d0", width=1)
    draw.line((0, cy, width, cy), fill="#c0c9ca", width=1)
    draw.line((cx, 0, cx, height), fill="#c0c9ca", width=1)

    def xy(point) -> Tuple[float, float]:
        return cx + float(point[0]) * scale, cy - float(point[1]) * scale

    points = np.asarray(frame.points)
    if len(points):
        stride = max(1, len(points) // 22_000)
        sampled = points[::stride]
        low = float(np.percentile(sampled[:, 2], 5))
        high = float(np.percentile(sampled[:, 2], 95))
        span = max(0.1, high - low)
        for point in sampled:
            px, py = xy(point)
            if 0 <= px < width and 0 <= py < height:
                t = max(0.0, min(1.0, (float(point[2]) - low) / span))
                color = (int(32 + 25 * t), int(62 + 120 * t), int(76 + 130 * t))
                draw.point((px, py), fill=color)

    if show_tracks:
        for track in frame.tracks:
            if len(track.points) >= 2:
                draw.line([xy(point) for point in track.points], fill="#7c4dff", width=2)
    for box in frame.boxes:
        target = (target_colors or {}).get(box.token)
        if show_boxes or target:
            corners = np.asarray(box.corners)
            color = target or CATEGORY_COLORS[category_group(box.category)]
            polygon = [xy(corners[:, index]) for index in (0, 1, 2, 3, 0)]
            draw.line(polygon, fill=color, width=6 if target else 2)

    ego = [(cx + 9, cy), (cx - 7, cy - 6), (cx - 7, cy + 6)]
    draw.polygon(ego, fill="#5c6670", outline=PAPER_INK)
    return image


def render_annotated_camera(
    repository: NuScenesSceneRepository,
    frame: FrameData,
    camera: str,
    show_boxes: bool = True,
    target_colors: Optional[Mapping[str, str]] = None,
) -> Image.Image:
    """Return a camera image with nuScenes boxes projected through calibration."""
    image = repository.camera_image(frame, camera)
    if not show_boxes and not target_colors:
        return image
    try:
        sample = repository.nusc.get("sample", frame.sample_token)
        camera_token = sample.get("data", {}).get(camera)
        if not camera_token:
            return image
        _, boxes, intrinsic = repository.nusc.get_sample_data(camera_token)
        matrix = np.asarray(intrinsic, dtype=np.float64)
        if matrix.shape != (3, 3):
            return image
        draw = ImageDraw.Draw(image)
        for box in boxes:
            target = (target_colors or {}).get(getattr(box, "token", ""))
            if not show_boxes and not target:
                continue
            corners = np.asarray(box.corners(), dtype=np.float64)
            if corners.shape != (3, 8):
                continue
            depth = corners[2]
            projected = matrix @ corners
            projected[:2] /= np.maximum(projected[2:3], 1e-6)
            color = target or CATEGORY_COLORS[category_group(str(getattr(box, "name", "")))]
            for start, end in BOX_EDGES:
                if depth[start] <= 0.1 or depth[end] <= 0.1:
                    continue
                a = tuple(projected[:2, start])
                b = tuple(projected[:2, end])
                draw.line((a, b), fill=color, width=4)
            # A tight rectangle makes the selected target legible after downsampling.
            visible = projected[:2, depth > 0.1]
            if target and visible.shape[1] and np.isfinite(visible).all():
                x0, y0 = np.maximum(visible.min(axis=1), (0, 0))
                x1, y1 = np.minimum(visible.max(axis=1), image.size)
                if x1 > x0 and y1 > y0:
                    draw.rectangle((x0, y0, x1, y1), outline=target, width=8)
        return image
    except Exception:
        # Custom/test datasets may not expose camera calibration or annotations.
        return image


def _wrapped_lines(draw, text: str, font, width: int) -> List[Tuple[int, int]]:
    """Return character spans; keep newlines and wrap English, Chinese and long IDs."""
    lines = []
    start = 0
    while start < len(text):
        end = start
        last_space = None
        while end < len(text) and text[end] != "\n":
            if draw.textlength(text[start:end + 1], font=font) > width and end > start:
                break
            if text[end].isspace():
                last_space = end
            end += 1
        if end < len(text) and text[end] != "\n" and last_space is not None:
            end = last_space
        lines.append((start, end))
        start = end + 1 if end < len(text) and text[end].isspace() else end
    return lines or [(0, 0)]


def _phrase_spans(text: str, phrases: Iterable[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase:
            continue
        for match in re.finditer(re.escape(phrase), text, flags=re.IGNORECASE):
            spans.append(match.span())
    return spans


def _draw_highlighted_answer(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    yellow_phrases: Sequence[str] = (),
    green_phrases: Sequence[str] = (),
    error_phrases: Sequence[str] = (),
    size: int = 23,
) -> None:
    """Use the same evidence colors in every answer; red marks user-selected errors."""
    x0, y0, x1, _ = box
    font = _font(size)
    yellow = _phrase_spans(text, yellow_phrases)
    green = _phrase_spans(text, green_phrases)
    errors = _phrase_spans(text, error_phrases)
    for row, (start, end) in enumerate(_wrapped_lines(draw, text, font, x1 - x0)):
        y = y0 + row * (size + 11)
        runs = []
        for index in range(start, end):
            background = (PAPER_YELLOW if any(a <= index < b for a, b in yellow) else
                          PAPER_GREEN if any(a <= index < b for a, b in green) else None)
            foreground = PAPER_ERROR if any(a <= index < b for a, b in errors) else PAPER_INK
            style = (background, foreground)
            if runs and runs[-1][2] == style:
                runs[-1] = (runs[-1][0], index + 1, style)
            else:
                runs.append((index, index + 1, style))
        for a, b, (background, foreground) in runs:
            x = x0 + draw.textlength(text[start:a], font=font)
            right = x0 + draw.textlength(text[start:b], font=font)
            if background:
                draw.rectangle((x, y + 5, right, y + size + 8), fill=background)
            draw.text((x, y), text[a:b], font=font, fill=foreground)


def compose_paper_case(
    front_images: Sequence[Image.Image],
    back_images: Sequence[Image.Image],
    bev_images: Sequence[Image.Image],
    frame_indices: Sequence[int],
    question: str,
    baseline_answer: str,
    b4dl_answer: str,
    baseline_label: str = "Baseline",
    b4dl_label: str = "B4DL model (Ours)",
    baseline_highlights: Sequence[str] = (),
    b4dl_highlights: Sequence[str] = (),
    title: str = "QUALITATIVE CASE STUDY",
    *,
    ground_truth: str = "",
    answer_panels: Optional[Sequence[AnswerPanel]] = None,
    yellow_phrases: Sequence[str] = (),
    green_phrases: Sequence[str] = (),
    layout: str = "comparison",
    conclusion: str = "",
    lidar_view: str = "3d",
    comparison_cases: Optional[Sequence[TemporalCase]] = None,
) -> Image.Image:
    """Figure 5 comparison / Figure 8 ablation, with independently supplied evidence."""
    columns = len(frame_indices)
    if columns < 2 or not (
        len(front_images) == len(back_images) == len(bev_images) == columns
    ):
        raise ValueError("三组视图必须与帧号数量一致，且至少包含 2 帧")

    if layout not in ("comparison", "ablation"):
        raise ValueError("版式必须为 comparison 或 ablation")
    if lidar_view not in ("3d", "bev"):
        raise ValueError("LiDAR 视图必须为 3d 或 bev")
    if comparison_cases:
        if layout != "comparison":
            raise ValueError("多问题对照使用 comparison 版式")
        return compose_temporal_cases(front_images, back_images, bev_images, frame_indices,
                                      comparison_cases, title, yellow_phrases, green_phrases,
                                      conclusion, lidar_view)
    panels = tuple(answer_panels) if answer_panels is not None else (
        AnswerPanel(baseline_label, baseline_answer), AnswerPanel(b4dl_label, b4dl_answer),
    )
    expected = 3 if layout == "ablation" else 2
    if len(panels) != expected:
        raise ValueError(f"{layout} 版式需要 {expected} 组模型答案")
    if not str(question or "").strip() or any(not str(p.answer or "").strip() for p in panels):
        raise ValueError("请填写问题与每组模型的实际答案")
    # Legacy arguments remain accepted, now applied consistently to all answers.
    yellow_phrases = tuple(yellow_phrases) + tuple(baseline_highlights)
    green_phrases = tuple(green_phrases) + tuple(b4dl_highlights)
    canvas_w = 2000
    margin = 32
    label_w = 126
    gap = 8
    content_w = canvas_w - margin * 2 - label_w
    cell_w = (content_w - gap * (columns - 1)) // columns
    cell_h = EVIDENCE_CELL_H
    top = EVIDENCE_TOP
    row_gap = EVIDENCE_GAP
    panel_gap = 10
    panel_w = (canvas_w - 2 * margin - panel_gap * (len(panels) - 1)) // len(panels)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    def text_height(text, width, size=23):
        return len(_wrapped_lines(measure, str(text or ""), _font(size), width)) * (size + 11)
    question_h = max(68, text_height(question, canvas_w - 2 * margin - 176) + 28)
    answer_label_h = max(text_height(p.label, panel_w - 36, 21) for p in panels) + 24
    answer_h = answer_label_h + max(text_height(p.answer, panel_w - 36) for p in panels) + 32
    gt_h = text_height(ground_truth, canvas_w - 2 * margin - 36) + 72 if ground_truth else 0
    conclusion_h = text_height(conclusion, canvas_w - 2 * margin - 36) + 72 if conclusion else 0
    rows = [("Front\nView", front_images), ("Back\nView", back_images),
            ("LiDAR\n3D" if lidar_view == "3d" else "LiDAR\nBEV", bev_images)]
    if layout == "ablation":
        rows.pop(1)
    evidence_bottom = top + len(rows) * cell_h + (len(rows) - 1) * row_gap
    canvas_h = evidence_bottom + 18 + question_h + gt_h + answer_h + conclusion_h + 64
    image = Image.new("RGB", (canvas_w, canvas_h), PAPER_BG)
    draw = ImageDraw.Draw(image)

    draw.text((margin, 8), title, font=_font(25, True), fill=PAPER_INK)
    arrow_y = 78
    arrow_x0 = margin + label_w
    arrow_x1 = canvas_w - margin
    draw.line((arrow_x0, arrow_y, arrow_x1, arrow_y), fill=PAPER_INK, width=4)
    draw.polygon(
        [(arrow_x1, arrow_y), (arrow_x1 - 18, arrow_y - 9), (arrow_x1 - 18, arrow_y + 9)],
        fill=PAPER_INK,
    )
    draw.text((arrow_x0 + (arrow_x1 - arrow_x0) / 2 - 30, 43), "Time", font=_font(22), fill=PAPER_INK)
    for row_index, (label, images) in enumerate(rows):
        y = top + row_index * (cell_h + row_gap)
        draw.multiline_text((margin, y + 58), label, font=_font(18, True), fill=PAPER_INK, spacing=2)
        draw.line((margin + label_w - 14, y, margin + label_w - 14, y + cell_h), fill=PAPER_LINE, width=2)
        for column, (source, frame_index) in enumerate(zip(images, frame_indices)):
            x = margin + label_w + column * (cell_w + gap)
            cell = _fit_cell(source, (cell_w, cell_h))
            image.paste(cell, (x, y))
            draw.rectangle((x, y, x + cell_w, y + cell_h), outline=PAPER_INK, width=2)
            draw.rectangle((x + 6, y + 6, x + 94, y + 33), fill="#ffffff")
            draw.text((x + 12, y + 5), f"Frame {frame_index:03d}", font=_font(16), fill=PAPER_INK)

    question_y = evidence_bottom + 18
    draw.rectangle((margin, question_y, canvas_w - margin, question_y + question_h),
                   outline=PAPER_INK, width=2)
    draw.text((margin + 18, question_y + 14), "Question:", font=_font(24, True), fill=PAPER_INK)
    _draw_highlighted_answer(draw, (margin + 176, question_y + 14, canvas_w - margin - 18, 0), question)

    def full_width_block(y, height, label, text, highlights=False):
        draw.rectangle((margin, y, canvas_w - margin, y + height), outline=PAPER_INK, width=2)
        draw.text((margin + 18, y + 10), label, font=_font(22, True), fill=PAPER_INK)
        _draw_highlighted_answer(draw, (margin + 18, y + 48, canvas_w - margin - 18, 0), text,
                                 yellow_phrases if highlights else (), green_phrases if highlights else ())

    answers_y = question_y + question_h
    if ground_truth:
        full_width_block(answers_y, gt_h, "Ground Truth", ground_truth, highlights=True)
        answers_y += gt_h
    for column, panel in enumerate(panels):
        x = margin + column * (panel_w + panel_gap)
        draw.rectangle((x, answers_y, x + panel_w, answers_y + answer_h), outline=PAPER_INK, width=2)
        _draw_highlighted_answer(draw, (x + 18, answers_y + 8, x + panel_w - 18, 0), panel.label, size=21)
        draw.line((x, answers_y + answer_label_h, x + panel_w, answers_y + answer_label_h), fill=PAPER_INK, width=1)
        _draw_highlighted_answer(draw, (x + 18, answers_y + answer_label_h + 12, x + panel_w - 18, 0),
                                 panel.answer, yellow_phrases, green_phrases, panel.error_phrases)
    footer_y = answers_y + answer_h
    if conclusion:
        full_width_block(footer_y, conclusion_h, "Observation / 观察结论", conclusion)
        footer_y += conclusion_h
    legend = [(PAPER_YELLOW, "Front target / 前方目标"), (PAPER_GREEN, "Rear target / 后方目标")]
    for column, (color, label) in enumerate(legend):
        x = margin + column * 440
        draw.rectangle((x, footer_y + 20, x + 22, footer_y + 42), fill=color, outline=PAPER_INK)
        draw.text((x + 32, footer_y + 15), label, font=_font(18), fill=PAPER_INK)
    draw.text((canvas_w - 555, footer_y + 15), "Red text: marked error / 红字：标记错误", font=_font(18), fill=PAPER_ERROR)
    return image


def compose_temporal_cases(fronts, backs, lidars, indices, cases, title,
                           yellow_phrases=(), green_phrases=(), conclusion="", lidar_view="3d"):
    """Several independent question/model pairs beneath one synchronized sequence."""
    if not 1 <= len(cases) <= 4:
        raise ValueError("时序图版需要 1 到 4 组问答对比")
    for case in cases:
        if len(case.answers) != 2 or not case.question.strip() or any(not a.answer.strip() for a in case.answers):
            raise ValueError("每组时序案例需要问题与两组模型的实际答案")
    first = cases[0]
    base = compose_paper_case(fronts, backs, lidars, indices, first.question,
                              first.answers[0].answer, first.answers[1].answer,
                              answer_panels=first.answers, title=title, lidar_view=lidar_view)
    evidence_h = EVIDENCE_TOP + 3 * EVIDENCE_CELL_H + 2 * EVIDENCE_GAP + 18
    margin, gap = 32, 20
    columns = min(2, len(cases))
    width = (base.width - 2 * margin - (columns - 1) * gap) // columns
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def height(text, available, size=23):
        return len(_wrapped_lines(measure, text, _font(size), available)) * (size + 11)

    sizes = []
    for case in cases:
        question_h = height(case.question, width - 40) + 62
        gt_h = height(case.ground_truth, width - 40) + 48 if case.ground_truth else 0
        panel_w = (width - 10) // 2
        labels_h = max(height(a.label, panel_w - 32, 21) for a in case.answers) + 20
        answers_h = max(height(a.answer, panel_w - 32) for a in case.answers) + labels_h + 24
        sizes.append((question_h, gt_h, labels_h, answers_h))
    row_heights = [max(sum(sizes[i]) for i in range(start, min(start + columns, len(cases)))) + 44
                   for start in range(0, len(cases), columns)]
    conclusion_h = height(conclusion, base.width - 2 * margin) + 66 if conclusion else 0
    image = Image.new("RGB", (base.width, evidence_h + sum(row_heights) + conclusion_h + 68), "white")
    image.paste(base.crop((0, 0, base.width, evidence_h)), (0, 0))
    draw = ImageDraw.Draw(image)
    y = evidence_h
    for row, row_h in enumerate(row_heights):
        for column in range(columns):
            index = row * columns + column
            if index >= len(cases):
                break
            case = cases[index]
            q_h, gt_h, labels_h, a_h = sizes[index]
            x = margin + column * (width + gap)
            draw.text((x, y), f"({chr(97 + index)})", font=_font(23, True), fill=PAPER_INK)
            block_y = y + 36
            draw.rectangle((x, block_y, x + width, block_y + q_h), outline=PAPER_LINE, width=2)
            draw.text((x + 16, block_y + 8), "Question", font=_font(21, True), fill=PAPER_INK)
            _draw_highlighted_answer(draw, (x + 16, block_y + 44, x + width - 16, 0), case.question)
            block_y += q_h
            if case.ground_truth:
                draw.text((x + 16, block_y + 4), "Ground Truth", font=_font(21, True), fill=PAPER_INK)
                _draw_highlighted_answer(draw, (x + 16, block_y + 40, x + width - 16, 0),
                                         case.ground_truth, yellow_phrases, green_phrases)
                block_y += gt_h
            panel_w = (width - 10) // 2
            for k, answer in enumerate(case.answers):
                left = x + k * (panel_w + 10)
                draw.rectangle((left, block_y, left + panel_w, block_y + a_h), outline=PAPER_LINE, width=2)
                _draw_highlighted_answer(draw, (left + 16, block_y + 6, left + panel_w - 16, 0),
                                         answer.label, size=21)
                draw.line((left, block_y + labels_h, left + panel_w, block_y + labels_h), fill=PAPER_LINE)
                _draw_highlighted_answer(draw, (left + 16, block_y + labels_h + 10, left + panel_w - 16, 0),
                                         answer.answer, yellow_phrases, green_phrases, answer.error_phrases)
        y += row_h
    if conclusion:
        draw.text((margin, y), "Observation / 观察结论", font=_font(22, True), fill=PAPER_INK)
        _draw_highlighted_answer(draw, (margin, y + 38, base.width - margin, 0), conclusion)
        y += conclusion_h
    for x, color, label in ((margin, PAPER_YELLOW, "Front target / 前方目标"),
                            (margin + 480, PAPER_GREEN, "Rear target / 后方目标")):
        draw.rectangle((x, y + 18, x + 22, y + 40), fill=color, outline=PAPER_LINE)
        draw.text((x + 32, y + 12), label, font=_font(18), fill=PAPER_MUTED)
    return image


def _save_artifact(image, identity, selected, output_dir, prefix="paper-case"):
    directory = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="b4dl-paper-case-"))
    directory.mkdir(parents=True, exist_ok=True)
    identity = re.sub(r"[^a-zA-Z0-9_-]+", "-", identity).strip("-") or "scene"
    png_path = directory / f"{prefix}-{identity}.png"
    pdf_path = directory / f"{prefix}-{identity}.pdf"
    image.save(png_path, format="PNG", dpi=(180, 180))
    image.save(pdf_path, format="PDF", resolution=180.0)
    return PaperCaseArtifact(image, str(png_path), str(pdf_path), tuple(selected))


def build_reasoning_board(repository, cases: Sequence[ReasoningCase], *,
                          title="3D QUESTION ANSWERING", output_dir=None,
                          azimuth=-55.0, elevation=28.0, range_m=45.0):
    """Reason3D-style Q/A cards and raw/highlighted pairs from nuScenes XYZ.

    Every row retains its own scene/frame/answer. Colored points are selected
    from nuScenes annotation boxes; no semantic segmentation is inferred.
    """
    if not 1 <= len(cases) <= 6:
        raise ValueError("三维问答图版需要 1 到 6 个案例")
    margin, gap, card_w, scene_w = 32, 20, 580, 658
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    rows = []
    for case in cases:
        if not case.question.strip() or not case.answer.strip():
            raise ValueError("每个三维案例必须包含问题和实际模型答案")
        if not case.target_instances:
            raise ValueError("请选择至少一个目标 instance，用于原始场景与高亮对照")
        scene = repository.get_scene(case.scene_token)
        if not 0 <= case.frame_index < len(scene.sample_tokens):
            raise ValueError("三维案例帧号超出场景范围")
        frame = repository.get_frame(case.scene_token, case.frame_index)
        instances = dict.fromkeys(case.target_instances, TARGET_PURPLE)
        colors = _frame_target_colors(repository, frame, instances)
        available = {box.token for box in frame.boxes}
        if len(colors) < len(instances) or not set(colors).issubset(available):
            raise ValueError("所选目标未出现在该帧的三维框中，请调整帧号或目标")
        for box in frame.boxes:
            if box.token in colors and not (np.abs(box.corners[:2]) <= range_m).all():
                raise ValueError("目标超出三维显示范围，请增大范围")
        def text_h(text, size=23):
            return len(_wrapped_lines(measure, text, _font(size), card_w - 68)) * (size + 11)
        label_h = text_h(case.task_label or "3D QA", 24) + 18
        q_h = text_h(case.question) + 48
        a_h = text_h(case.answer) + text_h(case.answer_label, 20) + 32
        gt_h = text_h(case.ground_truth, 20) + 38 if case.ground_truth else 0
        row_h = max(348, label_h + q_h + a_h + gt_h + 28)
        raw = render_lidar_scene(frame, (scene_w, 320), azimuth=azimuth,
                                 elevation=elevation, range_m=range_m)
        highlighted = render_lidar_scene(frame, (scene_w, 320), colors,
                                         azimuth=azimuth, elevation=elevation, range_m=range_m)
        rows.append((case, scene, label_h, q_h, a_h, gt_h, row_h, raw, highlighted))
    image = Image.new("RGB", (2000, 108 + sum(r[6] + 54 for r in rows) + 52), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 10), title, font=_font(28, True), fill=PAPER_INK)
    for x, label in ((margin, "Task / 问答"), (margin + card_w + gap, "Original 3D / 原始场景"),
                     (margin + card_w + 2 * gap + scene_w, "Target evidence / 目标高亮")):
        draw.text((x + 8, 64), label, font=_font(22, True), fill=PAPER_INK)
    y = 108
    for case, scene, label_h, q_h, a_h, gt_h, row_h, raw, highlighted in rows:
        draw.rounded_rectangle((margin, y, margin + card_w, y + row_h), radius=16,
                               outline=PAPER_LINE, width=2)
        _draw_highlighted_answer(draw, (margin + 20, y + 8, margin + card_w - 20, 0),
                                 case.task_label or "3D QA", size=24)
        bubble_y = y + label_h
        draw.rounded_rectangle((margin + 18, bubble_y, margin + card_w - 18, bubble_y + q_h),
                               radius=12, fill="#fbf7dd")
        draw.text((margin + 34, bubble_y + 6), "Question", font=_font(19, True), fill=PAPER_MUTED)
        _draw_highlighted_answer(draw, (margin + 34, bubble_y + 36, margin + card_w - 34, 0), case.question)
        bubble_y += q_h + 10
        draw.rounded_rectangle((margin + 18, bubble_y, margin + card_w - 18, bubble_y + a_h),
                               radius=12, fill="#e8f2fa")
        _draw_highlighted_answer(draw, (margin + 34, bubble_y + 6, margin + card_w - 34, 0),
                                 case.answer_label, size=20)
        label_lines = len(_wrapped_lines(draw, case.answer_label, _font(20), card_w - 68))
        _draw_highlighted_answer(draw, (margin + 34, bubble_y + label_lines * 31 + 12,
                                      margin + card_w - 34, 0), case.answer)
        if case.ground_truth:
            gt_y = bubble_y + a_h + 6
            draw.text((margin + 34, gt_y), "Ground Truth", font=_font(19, True), fill=PAPER_MUTED)
            _draw_highlighted_answer(draw, (margin + 34, gt_y + 30, margin + card_w - 34, 0),
                                     case.ground_truth, size=20)
        scene_y = y + (row_h - 320) // 2
        image.paste(raw, (margin + card_w + gap, scene_y))
        image.paste(highlighted, (margin + card_w + 2 * gap + scene_w, scene_y))
        caption = f"{scene.scene_id or scene.name} · Frame {case.frame_index:03d}"
        if case.sample_id:
            caption += f" · {case.sample_id}"
        _draw_highlighted_answer(draw, (margin, y + row_h + 6, 1968, 0), caption, size=18)
        y += row_h + 54
    draw.rectangle((margin, y + 6, margin + 22, y + 28), fill=TARGET_PURPLE)
    draw.text((margin + 34, y), "Purple: points inside selected annotation boxes / 紫色：所选标注框内点，非模型分割预测",
              font=_font(20), fill=PAPER_MUTED)
    return _save_artifact(image, "board", [c.frame_index for c in cases], output_dir, "paper-3d-qa")


def split_highlights(value: str) -> Tuple[str, ...]:
    return tuple(
        item.strip()
        for item in re.split(r"[,，;；\n]+", str(value or ""))
        if item.strip()
    )


def build_paper_case(
    repository: NuScenesSceneRepository,
    scene_token: str,
    frame_indices: Optional[Sequence[int]],
    question: str,
    baseline_answer: str,
    b4dl_answer: str,
    baseline_label: str = "Baseline",
    b4dl_label: str = "B4DL model (Ours)",
    baseline_highlights: Sequence[str] = (),
    b4dl_highlights: Sequence[str] = (),
    title: str = "QUALITATIVE CASE STUDY",
    show_boxes: bool = True,
    show_tracks: bool = True,
    output_dir: Optional[str] = None,
    *,
    ground_truth: str = "",
    answer_panels: Optional[Sequence[AnswerPanel]] = None,
    yellow_phrases: Sequence[str] = (),
    green_phrases: Sequence[str] = (),
    front_instance: Optional[str] = None,
    rear_instance: Optional[str] = None,
    layout: str = "comparison",
    conclusion: str = "",
    lidar_view: str = "3d",
    comparison_cases: Optional[Sequence[TemporalCase]] = None,
    azimuth: float = -55.0,
    elevation: float = 28.0,
    range_m: float = 45.0,
) -> PaperCaseArtifact:
    """Load synchronized scene evidence and write PNG/PDF artifacts."""
    scene = repository.get_scene(scene_token)
    selected = (
        select_frame_indices(len(scene.sample_tokens), requested=frame_indices)
        if frame_indices is not None
        else select_frame_indices(len(scene.sample_tokens))
    )
    fronts: List[Image.Image] = []
    backs: List[Image.Image] = []
    bevs: List[Image.Image] = []
    if front_instance and front_instance == rear_instance:
        raise ValueError("前方和后方目标不能选择同一个 instance")
    instances = {}
    if front_instance:
        instances[front_instance] = PAPER_YELLOW
    if rear_instance:
        instances[rear_instance] = PAPER_GREEN
    seen_colors = set()
    for index in selected:
        frame = repository.get_frame(scene_token, index)
        colors = _frame_target_colors(repository, frame, instances) if instances else {}
        seen_colors.update(colors.values())
        fronts.append(render_annotated_camera(repository, frame, "CAM_FRONT", show_boxes, colors))
        backs.append(render_annotated_camera(repository, frame, "CAM_BACK", show_boxes, colors))
        if lidar_view == "3d":
            bevs.append(render_lidar_scene(frame, target_colors=colors, show_boxes=show_boxes,
                                           show_tracks=show_tracks, azimuth=azimuth,
                                           elevation=elevation, range_m=range_m))
        elif lidar_view == "bev":
            bevs.append(render_bev_image(frame, show_boxes=show_boxes, show_tracks=show_tracks,
                                         target_colors=colors))
        else:
            raise ValueError("LiDAR 视图必须为 3d 或 bev")
    if set(instances.values()) - seen_colors:
        raise ValueError("所选目标未出现在这些帧的标注中，请调整帧号或目标")
    image = compose_paper_case(
        fronts, backs, bevs, selected, question, baseline_answer, b4dl_answer,
        baseline_label=baseline_label, b4dl_label=b4dl_label,
        baseline_highlights=baseline_highlights, b4dl_highlights=b4dl_highlights,
        title=title,
        ground_truth=ground_truth, answer_panels=answer_panels,
        yellow_phrases=yellow_phrases, green_phrases=green_phrases,
        layout=layout, conclusion=conclusion, lidar_view=lidar_view,
        comparison_cases=comparison_cases,
    )
    return _save_artifact(image, scene.scene_id or scene.name, selected, output_dir)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 B4DL 论文风格定性案例图")
    parser.add_argument("--nuscenes-root", required=True)
    parser.add_argument("--scene-token")
    parser.add_argument("--scene-metadata")
    parser.add_argument("--nuscenes-version", default="v1.0-trainval")
    parser.add_argument("--frames", default="", help="逗号分隔帧号；留空自动均匀选择 5 帧")
    parser.add_argument("--question", default="")
    parser.add_argument("--baseline-answer", default="")
    parser.add_argument("--b4dl-answer", default="")
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--b4dl-label", default="B4DL model (Ours)")
    parser.add_argument("--baseline-highlights", default="")
    parser.add_argument("--b4dl-highlights", default="")
    parser.add_argument("--title", default="QUALITATIVE CASE STUDY")
    parser.add_argument("--layout", choices=("comparison", "ablation", "reasoning"), default="comparison")
    parser.add_argument("--cases-json", help="多案例文件；时序图包含 question/answers，三维图包含场景、帧、问题、答案与目标")
    parser.add_argument("--lidar-view", choices=("3d", "bev"), default="3d")
    parser.add_argument("--azimuth", type=float, default=-55)
    parser.add_argument("--elevation", type=float, default=28)
    parser.add_argument("--range-m", type=float, default=45)
    parser.add_argument("--ground-truth", default="")
    parser.add_argument("--yellow-phrases", default="", help="前方目标证据，应用于所有答案")
    parser.add_argument("--green-phrases", default="", help="后方目标证据，应用于所有答案")
    parser.add_argument("--baseline-errors", default="", help="基线答案中需标红的错误短语")
    parser.add_argument("--b4dl-errors", default="")
    parser.add_argument("--middle-label", default="B4DL without Metatoken")
    parser.add_argument("--middle-answer", default="", help="消融版式的第二组答案")
    parser.add_argument("--middle-errors", default="")
    parser.add_argument("--front-instance", help="前方目标的 nuScenes instance_token")
    parser.add_argument("--rear-instance", help="后方目标的 nuScenes instance_token")
    parser.add_argument("--conclusion", default="", help="人工填写的观察结论；留空不生成")
    parser.add_argument("--output-dir", default="paper_cases")
    parser.add_argument("--no-boxes", action="store_true")
    parser.add_argument("--no-tracks", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    repository = NuScenesSceneRepository(
        args.nuscenes_root,
        version=args.nuscenes_version,
        scene_metadata=args.scene_metadata,
    )
    raw_cases = json.loads(Path(args.cases_json).read_text(encoding="utf-8")) if args.cases_json else None
    if args.layout == "reasoning":
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError("三维问答导出需通过 --cases-json 提供案例列表")
        cases = [ReasoningCase(**{**row, "target_instances": tuple(row.get("target_instances", ()))})
                 for row in raw_cases]
        artifact = build_reasoning_board(repository, cases, title=args.title,
                                         output_dir=args.output_dir, azimuth=args.azimuth,
                                         elevation=args.elevation, range_m=args.range_m)
        print(artifact.png_path)
        print(artifact.pdf_path)
        return
    if not args.scene_token:
        raise ValueError("时序或消融导出需提供 --scene-token")
    scene = repository.get_scene(args.scene_token)
    frames = parse_frame_indices(args.frames, len(scene.sample_tokens))
    panels = [AnswerPanel(args.baseline_label, args.baseline_answer, split_highlights(args.baseline_errors))]
    if args.layout == "ablation":
        panels.append(AnswerPanel(args.middle_label, args.middle_answer, split_highlights(args.middle_errors)))
    panels.append(AnswerPanel(args.b4dl_label, args.b4dl_answer, split_highlights(args.b4dl_errors)))
    cases = None
    if raw_cases is not None:
        if args.layout != "comparison" or not isinstance(raw_cases, list):
            raise ValueError("多问题案例列表仅用于 comparison 版式")
        cases = [TemporalCase(row["question"], tuple(AnswerPanel(**answer) for answer in row["answers"]),
                              row.get("ground_truth", "")) for row in raw_cases]
    artifact = build_paper_case(
        repository, args.scene_token, frames, args.question,
        args.baseline_answer, args.b4dl_answer,
        baseline_label=args.baseline_label,
        b4dl_label=args.b4dl_label,
        baseline_highlights=split_highlights(args.baseline_highlights),
        b4dl_highlights=split_highlights(args.b4dl_highlights),
        title=args.title,
        show_boxes=not args.no_boxes,
        show_tracks=not args.no_tracks,
        output_dir=args.output_dir,
        ground_truth=args.ground_truth, answer_panels=panels,
        yellow_phrases=split_highlights(args.yellow_phrases),
        green_phrases=split_highlights(args.green_phrases),
        front_instance=args.front_instance, rear_instance=args.rear_instance,
        layout=args.layout, conclusion=args.conclusion,
        lidar_view=args.lidar_view, comparison_cases=cases,
        azimuth=args.azimuth, elevation=args.elevation, range_m=args.range_m,
    )
    print(artifact.png_path)
    print(artifact.pdf_path)


if __name__ == "__main__":
    main()
