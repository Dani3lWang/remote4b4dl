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
from typing import Iterable, List, Optional, Sequence, Tuple

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


PAPER_BG = "#f3f0e8"
PAPER_PANEL = "#fffdf8"
PAPER_INK = "#172028"
PAPER_MUTED = "#66727c"
PAPER_LINE = "#9da6ac"
PAPER_YELLOW = "#ffe84a"
PAPER_GREEN = "#70f08b"
PAPER_CYAN = "#00a7a0"


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
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)


def render_bev_image(
    frame: FrameData,
    size: Tuple[int, int] = (640, 320),
    range_m: float = 52.0,
    show_boxes: bool = True,
    show_tracks: bool = True,
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
    if show_boxes:
        for box in frame.boxes:
            corners = np.asarray(box.corners)
            color = CATEGORY_COLORS[category_group(box.category)]
            polygon = [xy(corners[:, index]) for index in (0, 1, 2, 3, 0)]
            draw.line(polygon, fill=color, width=3)

    ego = [(cx + 9, cy), (cx - 7, cy - 6), (cx - 7, cy + 6)]
    draw.polygon(ego, fill=PAPER_YELLOW, outline=PAPER_INK)
    return image


def render_annotated_camera(
    repository: NuScenesSceneRepository,
    frame: FrameData,
    camera: str,
    show_boxes: bool = True,
) -> Image.Image:
    """Return a camera image with nuScenes boxes projected through calibration."""
    image = repository.camera_image(frame, camera)
    if not show_boxes:
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
            corners = np.asarray(box.corners(), dtype=np.float64)
            if corners.shape != (3, 8):
                continue
            depth = corners[2]
            projected = matrix @ corners
            projected[:2] /= np.maximum(projected[2:3], 1e-6)
            color = CATEGORY_COLORS[category_group(str(getattr(box, "name", "")))]
            for start, end in BOX_EDGES:
                if depth[start] <= 0.1 or depth[end] <= 0.1:
                    continue
                a = tuple(projected[:2, start])
                b = tuple(projected[:2, end])
                draw.line((a, b), fill=color, width=4)
        return image
    except Exception:
        # Custom/test datasets may not expose camera calibration or annotations.
        return image


def _draw_fitted_text(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    max_size: int,
    min_size: int = 15,
    bold: bool = False,
    fill: str = PAPER_INK,
    spacing: int = 5,
) -> None:
    x0, y0, x1, y1 = box
    for size in range(max_size, min_size - 1, -1):
        font = _font(size, bold)
        words = str(text or "").split()
        lines: List[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if draw.textlength(candidate, font=font) <= x1 - x0 or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        line_height = size + spacing
        if line_height * max(1, len(lines)) <= y1 - y0:
            draw.multiline_text((x0, y0), "\n".join(lines), font=font, fill=fill, spacing=spacing)
            return
    draw.text((x0, y0), str(text or "")[:180], font=_font(min_size, bold), fill=fill)


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
    phrases: Sequence[str],
    highlight: str,
) -> None:
    """Draw wrapped text while preserving configured phrase highlights."""
    x0, y0, x1, y1 = box
    font = _font(23)
    line_height = 32
    spans = _phrase_spans(text, phrases)
    tokens = list(re.finditer(r"\S+\s*", text))
    x, y = x0, y0
    for token in tokens:
        value = token.group(0)
        width = float(draw.textlength(value, font=font))
        if x > x0 and x + width > x1:
            x, y = x0, y + line_height
        if y + line_height > y1:
            draw.text((x, y), "…", font=font, fill=PAPER_MUTED)
            break
        marked = any(token.start() < end and token.end() > start for start, end in spans)
        if marked:
            draw.rounded_rectangle(
                (x - 2, y + 1, x + width + 1, y + 27), radius=3, fill=highlight
            )
        draw.text((x, y), value, font=font, fill=PAPER_INK)
        x += width


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
) -> Image.Image:
    """Compose synchronized views and answer comparison into one figure."""
    columns = len(frame_indices)
    if columns < 2 or not (
        len(front_images) == len(back_images) == len(bev_images) == columns
    ):
        raise ValueError("三组视图必须与帧号数量一致，且至少包含 2 帧")

    canvas_w = 2000
    margin = 34
    label_w = 138
    gap = 12
    content_w = canvas_w - margin * 2 - label_w
    cell_w = (content_w - gap * (columns - 1)) // columns
    cell_h = 176
    top = 105
    row_gap = 16
    question_h = 82
    answer_h = 250
    canvas_h = top + 3 * cell_h + 2 * row_gap + question_h + answer_h + 86
    image = Image.new("RGB", (canvas_w, canvas_h), PAPER_BG)
    draw = ImageDraw.Draw(image)

    draw.text((margin, 24), title.upper(), font=_font(25, True), fill=PAPER_INK)
    draw.text((margin, 57), "SYNCHRONIZED SENSOR EVIDENCE", font=_font(13, True), fill=PAPER_CYAN)
    arrow_y = 61
    arrow_x0 = margin + label_w
    arrow_x1 = canvas_w - margin
    draw.line((arrow_x0, arrow_y, arrow_x1, arrow_y), fill=PAPER_INK, width=4)
    draw.polygon(
        [(arrow_x1, arrow_y), (arrow_x1 - 18, arrow_y - 9), (arrow_x1 - 18, arrow_y + 9)],
        fill=PAPER_INK,
    )
    draw.text((arrow_x0 + (arrow_x1 - arrow_x0) / 2 - 35, 28), "TIME", font=_font(15, True), fill=PAPER_INK)

    rows = (
        ("FRONT\nVIEW", front_images),
        ("BACK\nVIEW", back_images),
        ("LiDAR\nBEV", bev_images),
    )
    for row_index, (label, images) in enumerate(rows):
        y = top + row_index * (cell_h + row_gap)
        draw.multiline_text((margin, y + 58), label, font=_font(18, True), fill=PAPER_INK, spacing=2)
        draw.line((margin + label_w - 14, y, margin + label_w - 14, y + cell_h), fill=PAPER_LINE, width=2)
        for column, (source, frame_index) in enumerate(zip(images, frame_indices)):
            x = margin + label_w + column * (cell_w + gap)
            cell = _fit_cell(source, (cell_w, cell_h))
            image.paste(cell, (x, y))
            draw.rectangle((x, y, x + cell_w, y + cell_h), outline=PAPER_INK, width=2)
            draw.rectangle((x + 8, y + 8, x + 73, y + 32), fill=PAPER_INK)
            draw.text((x + 15, y + 10), f"F{frame_index:03d}", font=_font(13, True), fill="#ffffff")

    question_y = top + 3 * cell_h + 2 * row_gap + 22
    draw.rounded_rectangle(
        (margin, question_y, canvas_w - margin, question_y + question_h - 12),
        radius=8, fill=PAPER_PANEL, outline=PAPER_INK, width=2,
    )
    draw.rectangle((margin, question_y, margin + 158, question_y + question_h - 12), fill=PAPER_INK)
    draw.text((margin + 22, question_y + 22), "QUESTION", font=_font(17, True), fill="#ffffff")
    _draw_fitted_text(
        draw,
        (margin + 181, question_y + 17, canvas_w - margin - 18, question_y + question_h - 18),
        question, max_size=25, min_size=17,
    )

    answers_y = question_y + question_h
    panel_gap = 18
    panel_w = (canvas_w - 2 * margin - panel_gap) // 2
    panels = (
        (baseline_label, baseline_answer, baseline_highlights, PAPER_YELLOW),
        (b4dl_label, b4dl_answer, b4dl_highlights, PAPER_GREEN),
    )
    for column, (label, answer, phrases, color) in enumerate(panels):
        x = margin + column * (panel_w + panel_gap)
        draw.rounded_rectangle(
            (x, answers_y, x + panel_w, answers_y + answer_h),
            radius=10, fill=PAPER_PANEL, outline=PAPER_INK, width=2,
        )
        draw.rectangle((x, answers_y, x + 13, answers_y + answer_h), fill=color)
        draw.text((x + 34, answers_y + 22), label, font=_font(21, True), fill=PAPER_INK)
        draw.line((x + 34, answers_y + 56, x + panel_w - 24, answers_y + 56), fill=PAPER_LINE, width=1)
        _draw_highlighted_answer(
            draw,
            (x + 34, answers_y + 79, x + panel_w - 30, answers_y + answer_h - 22),
            str(answer or ""), phrases, color,
        )

    footer_y = answers_y + answer_h + 22
    draw.text((margin, footer_y), "B4DL · 4D LIDAR LANGUAGE MODEL", font=_font(13, True), fill=PAPER_CYAN)
    frame_text = "  /  ".join(f"FRAME {value:03d}" for value in frame_indices)
    footer_width = draw.textlength(frame_text, font=_font(12))
    draw.text((canvas_w - margin - footer_width, footer_y), frame_text, font=_font(12), fill=PAPER_MUTED)
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
    for index in selected:
        frame = repository.get_frame(scene_token, index)
        fronts.append(render_annotated_camera(repository, frame, "CAM_FRONT", show_boxes))
        backs.append(render_annotated_camera(repository, frame, "CAM_BACK", show_boxes))
        bevs.append(render_bev_image(frame, show_boxes=show_boxes, show_tracks=show_tracks))
    image = compose_paper_case(
        fronts, backs, bevs, selected, question, baseline_answer, b4dl_answer,
        baseline_label=baseline_label, b4dl_label=b4dl_label,
        baseline_highlights=baseline_highlights, b4dl_highlights=b4dl_highlights,
        title=title,
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
    )
    print(artifact.png_path)
    print(artifact.pdf_path)


if __name__ == "__main__":
    main()
