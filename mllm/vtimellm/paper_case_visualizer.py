"""Publication-ready qualitative case figures for B4DL.

The renderer is deliberately independent from Gradio and the model stack.  It
loads already indexed nuScenes frames, projects available 3D annotations into
front/back cameras, draws a static LiDAR BEV, and composes a paper-style figure
that can be exported as PNG or PDF.
"""

from __future__ import annotations

import argparse
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

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
) -> Image.Image:
    """Figure 5 comparison / Figure 8 ablation, with independently supplied evidence."""
    columns = len(frame_indices)
    if columns < 2 or not (
        len(front_images) == len(back_images) == len(bev_images) == columns
    ):
        raise ValueError("三组视图必须与帧号数量一致，且至少包含 2 帧")

    if layout not in ("comparison", "ablation"):
        raise ValueError("版式必须为 comparison 或 ablation")
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
    cell_h = 196
    top = 110
    row_gap = 10
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
    rows = [("Front\nView", front_images), ("Back\nView", back_images), ("LiDAR\nBEV", bev_images)]
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
        bevs.append(render_bev_image(frame, show_boxes=show_boxes, show_tracks=show_tracks,
                                     target_colors=colors))
    if set(instances.values()) - seen_colors:
        raise ValueError("所选目标未出现在这些帧的标注中，请调整帧号或目标")
    image = compose_paper_case(
        fronts, backs, bevs, selected, question, baseline_answer, b4dl_answer,
        baseline_label=baseline_label, b4dl_label=b4dl_label,
        baseline_highlights=baseline_highlights, b4dl_highlights=b4dl_highlights,
        title=title,
        ground_truth=ground_truth, answer_panels=answer_panels,
        yellow_phrases=yellow_phrases, green_phrases=green_phrases,
        layout=layout, conclusion=conclusion,
    )
    directory = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="b4dl-paper-case-"))
    directory.mkdir(parents=True, exist_ok=True)
    identity = re.sub(r"[^a-zA-Z0-9_-]+", "-", scene.scene_id or scene.name).strip("-") or "scene"
    png_path = directory / f"paper-case-{identity}.png"
    pdf_path = directory / f"paper-case-{identity}.pdf"
    image.save(png_path, format="PNG", dpi=(180, 180))
    image.save(pdf_path, format="PDF", resolution=180.0)
    return PaperCaseArtifact(image, str(png_path), str(pdf_path), selected)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 B4DL 论文风格定性案例图")
    parser.add_argument("--nuscenes-root", required=True)
    parser.add_argument("--scene-token", required=True)
    parser.add_argument("--scene-metadata")
    parser.add_argument("--nuscenes-version", default="v1.0-trainval")
    parser.add_argument("--frames", default="", help="逗号分隔帧号；留空自动均匀选择 5 帧")
    parser.add_argument("--question", required=True)
    parser.add_argument("--baseline-answer", required=True)
    parser.add_argument("--b4dl-answer", required=True)
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--b4dl-label", default="B4DL model (Ours)")
    parser.add_argument("--baseline-highlights", default="")
    parser.add_argument("--b4dl-highlights", default="")
    parser.add_argument("--title", default="QUALITATIVE CASE STUDY")
    parser.add_argument("--layout", choices=("comparison", "ablation"), default="comparison")
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
    scene = repository.get_scene(args.scene_token)
    frames = parse_frame_indices(args.frames, len(scene.sample_tokens))
    panels = [AnswerPanel(args.baseline_label, args.baseline_answer, split_highlights(args.baseline_errors))]
    if args.layout == "ablation":
        panels.append(AnswerPanel(args.middle_label, args.middle_answer, split_highlights(args.middle_errors)))
    panels.append(AnswerPanel(args.b4dl_label, args.b4dl_answer, split_highlights(args.b4dl_errors)))
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
    )
    print(artifact.png_path)
    print(artifact.pdf_path)


if __name__ == "__main__":
    main()
