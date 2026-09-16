"""B4DL 4D LiDAR scene explorer, evaluation dashboard, and model chat UI.

Viewer-only mode needs nuScenes plus the demo requirements. Model chat is
enabled only when all model/feature arguments are supplied as a complete set.
Evaluation mode reads predictions/metrics without loading model weights.
No training module or training dataset is imported by this entrypoint.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


_SCRIPT_DIR = Path(__file__).resolve().parents[1]  # mllm/
_REPO_ROOT = _SCRIPT_DIR.parent
_MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_MODULE_DIR))

from lidar_visualizer import (  # noqa: E402
    CAMERA_VIEWS,
    NuScenesSceneRepository,
    SceneRef,
    advance_playback,
    empty_plotly_figure,
    make_plotly_figures,
    step_frame,
    validate_scene_features,
)
from model_effects import (  # noqa: E402
    TASK_LABELS,
    TASKS,
    EvaluationRepository,
)
from effect_visualizer import (  # noqa: E402
    confusion_matrix_figure,
    empty_effect_figure,
    metric_cards_html,
    overview_metrics_figure,
    per_task_metrics_figure,
    sample_diagnosis_html,
    time_grounding_diagnostics_figure,
    timeline_figure,
)
from paper_case_visualizer import (  # noqa: E402
    build_paper_case,
    parse_frame_indices,
    select_frame_indices,
    split_highlights,
)


APP_CSS = """
:root {
  --b4-bg: #060b10;
  --b4-panel: #0b131c;
  --b4-line: #263746;
  --b4-text: #dce7f2;
  --b4-muted: #8093a7;
  --b4-cyan: #00d4c7;
  --b4-amber: #ffb000;
}
body, .gradio-container {
  background:
    linear-gradient(rgba(38,55,70,.11) 1px, transparent 1px),
    linear-gradient(90deg, rgba(38,55,70,.11) 1px, transparent 1px),
    radial-gradient(circle at 18% 4%, rgba(0,212,199,.09), transparent 32%),
    var(--b4-bg) !important;
  background-size: 32px 32px, 32px 32px, auto, auto !important;
  color: var(--b4-text) !important;
  font-family: "Bahnschrift", "DIN Alternate", sans-serif !important;
}
.gradio-container { max-width: 1920px !important; padding: 18px 22px 28px !important; }
#b4-hero {
  position: relative; overflow: hidden; border: 1px solid var(--b4-line);
  background: linear-gradient(115deg, rgba(16,27,38,.98), rgba(7,14,21,.96));
  padding: 20px 24px; margin-bottom: 14px; box-shadow: 0 18px 70px rgba(0,0,0,.28);
}
#b4-hero::after {
  content: ""; position: absolute; inset: 0 0 0 auto; width: 32%;
  background: repeating-linear-gradient(125deg, transparent 0 18px, rgba(0,212,199,.10) 19px 20px);
  pointer-events: none;
}
.b4-kicker { color: var(--b4-cyan); letter-spacing: .22em; font-size: 11px; font-weight: 700; }
.b4-title { margin: 5px 0 3px; font-size: clamp(25px, 3vw, 45px); line-height: 1; letter-spacing: -.035em; }
.b4-subtitle { color: var(--b4-muted); font-size: 13px; letter-spacing: .045em; }
.b4-live { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--b4-cyan); box-shadow: 0 0 15px var(--b4-cyan); margin-right: 8px; animation: b4pulse 2.4s ease-in-out infinite; }
@keyframes b4pulse { 50% { opacity: .35; transform: scale(.72); } }
#b4-controls, #b4-viewer, #b4-chat {
  border: 1px solid var(--b4-line) !important; background: rgba(11,19,28,.93) !important;
  box-shadow: 0 14px 44px rgba(0,0,0,.20); padding: 12px !important;
}
#b4-controls { border-top: 3px solid var(--b4-amber) !important; }
#b4-viewer { border-top: 3px solid var(--b4-cyan) !important; }
#b4-chat { border-top: 3px solid #ff4d6d !important; }
#b4-frame-summary { border-left: 2px solid var(--b4-cyan); padding-left: 11px; color: var(--b4-muted); }
#b4-status { min-height: 52px; color: var(--b4-muted); }
button.primary { background: var(--b4-cyan) !important; color: #031014 !important; border: none !important; font-weight: 800 !important; }
button.secondary { border-color: var(--b4-line) !important; }
.tabs > .tab-nav { border-bottom: 1px solid var(--b4-line) !important; }
.tabs > .tab-nav button.selected { color: var(--b4-cyan) !important; border-bottom-color: var(--b4-cyan) !important; }
textarea, input { font-family: "Aptos", "Segoe UI", sans-serif !important; }
.effect-shell {
  border: 1px solid var(--b4-line); border-top: 3px solid var(--b4-cyan);
  background: rgba(8,16,24,.94); padding: 14px; margin-bottom: 16px;
  box-shadow: 0 18px 58px rgba(0,0,0,.28);
}
.effect-heading { margin: 0 0 11px; color: var(--b4-text); font-size: 18px; letter-spacing: .08em; }
.metric-rail { display: grid; grid-template-columns: repeat(7, minmax(112px, 1fr)); gap: 8px; margin: 4px 0 14px; }
.metric-card {
  min-height: 78px; border: 1px solid var(--b4-line); padding: 11px 12px;
  background: linear-gradient(145deg, rgba(0,212,199,.09), rgba(11,19,28,.95));
  position: relative; overflow: hidden;
}
.metric-card::after { content: ""; position: absolute; left: 0; bottom: 0; width: 100%; height: 2px; background: var(--b4-cyan); }
.metric-card span { display: block; color: var(--b4-muted); font-size: 10px; letter-spacing: .15em; }
.metric-card strong { display: block; margin-top: 8px; color: var(--b4-text); font-size: 24px; font-weight: 600; }
.metric-card.is-na strong { color: var(--b4-muted); }
.effect-warning { border-left: 2px solid var(--b4-amber); padding: 8px 12px; margin: 4px 0 12px; color: var(--b4-amber); background: rgba(255,176,0,.06); }
.diagnosis-strip { display: flex; flex-wrap: wrap; gap: 7px; margin: 6px 0 10px; }
.diagnosis-strip span { border: 1px solid var(--b4-line); padding: 7px 10px; background: #08111a; color: var(--b4-muted); }
.diagnosis-strip span:first-child { color: var(--b4-cyan); border-color: rgba(0,212,199,.45); }
#effect-sample-panel { border-left: 3px solid var(--b4-amber); padding-left: 14px; }
#paper-case-builder {
  border: 1px solid var(--b4-line); border-left: 3px solid var(--b4-amber);
  background: linear-gradient(145deg, rgba(255,176,0,.055), rgba(8,16,24,.96));
  padding: 14px !important;
}
#paper-case-preview {
  border: 1px solid var(--b4-line); background: #f3f0e8; padding: 8px !important;
}
@media (max-width: 1050px) { .metric-rail { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 620px) { .metric-rail { grid-template-columns: repeat(2, 1fr); } }
"""

HERO_HTML = """
<section id="b4-hero">
  <div class="b4-kicker"><span class="b4-live"></span>B4DL / SENSOR OPERATIONS</div>
  <h1 class="b4-title">4D LiDAR 模型诊断台</h1>
  <div class="b4-subtitle">MODEL EFFECTS · TEMPORAL ERROR · 3D / BEV · SCENE CHAT</div>
</section>
"""


def _model_option_state(args: argparse.Namespace) -> Tuple[bool, List[str]]:
    required = {
        "model_base": args.model_base,
        "pretrain_mm_mlp_adapter": args.pretrain_mm_mlp_adapter,
        "stage2": args.stage2,
        "feat_folder": args.feat_folder,
    }
    provided = [name for name, value in required.items() if value]
    if not provided:
        if args.stage3:
            return False, ["stage3 不能在未配置基础模型时单独使用"]
        return False, []
    missing = [name for name, value in required.items() if not value]
    return not missing, missing


class OptionalInferenceEngine:
    """One model instance with bounded feature cache and serialized generation."""

    def __init__(self, args: argparse.Namespace) -> None:
        enabled, missing = _model_option_state(args)
        if missing:
            raise ValueError("模型模式参数必须成套提供，缺少：" + ", ".join(missing))
        self.enabled = enabled
        self.feat_folder = Path(args.feat_folder).expanduser().resolve() if enabled else None
        self.device = None
        self.tokenizer = None
        self.model = None
        self._lock = threading.Lock()
        self._feature_cache: "OrderedDict[str, object]" = OrderedDict()
        self._feature_cache_size = 6
        if not enabled:
            return
        if not self.feat_folder.is_dir():
            raise FileNotFoundError(f"特征目录不存在：{self.feat_folder}")

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("模型模式需要 CUDA；若只查看数据，请移除全部模型参数")
        from vtimellm.model.builder import load_pretrained_model
        from vtimellm.utils import disable_torch_init

        self.device = f"cuda:{args.gpu_id}"
        disable_torch_init()
        self.tokenizer, self.model, _ = load_pretrained_model(args, args.stage2, args.stage3)
        self.model = self.model.to(torch.float16).to(self.device)
        self.model.eval()

    def _feature_path(self, scene: SceneRef) -> Path:
        if not scene.scene_id:
            raise ValueError("当前场景缺少 B4DL scene_id 映射")
        return self.feat_folder / f"{scene.scene_id}.npy"

    def _load_features(self, scene: SceneRef):
        import torch
        key = scene.scene_token
        cached = self._feature_cache.get(key)
        if cached is not None:
            self._feature_cache.move_to_end(key)
            return cached
        path = self._feature_path(scene)
        if not path.is_file():
            raise FileNotFoundError(f"缺少场景特征：{path}")
        values = np.load(path, allow_pickle=False)
        validate_scene_features(values, len(scene.sample_tokens))
        tensor = torch.from_numpy(values).to(torch.float16).to(self.device)
        self._feature_cache[key] = tensor
        self._feature_cache.move_to_end(key)
        while len(self._feature_cache) > self._feature_cache_size:
            self._feature_cache.popitem(last=False)
        return tensor

    def scene_status(self, scene: SceneRef) -> Tuple[bool, str]:
        if not self.enabled:
            return False, "查看模式 · 未加载语言模型"
        try:
            self._load_features(scene)
        except Exception as exc:
            return False, f"问答不可用 · {exc}"
        return True, "模型就绪 · 使用整场景 LiDAR 特征"

    def answer(self, scene: SceneRef, message: str, history, conversation):
        if not message or not message.strip():
            return history or [], conversation, "问题不能为空"
        ready, status = self.scene_status(scene)
        history = list(history or [])
        history.append({"role": "user", "content": message.strip()})
        if not ready:
            history.append({"role": "assistant", "content": status})
            return history, conversation, status

        import torch
        from vtimellm.constants import IMAGE_TOKEN_INDEX
        from vtimellm.conversation import SeparatorStyle, conv_templates
        from vtimellm.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token

        with self._lock, torch.inference_mode():
            features = self._load_features(scene)
            if conversation is None:
                conversation = conv_templates["v1"].copy()
                prompt_message = "<4DLiDAR>\n<video>\n" + message.strip()
            else:
                prompt_message = message.strip()
            conversation.append_message(conversation.roles[0], prompt_message)
            conversation.append_message(conversation.roles[1], None)
            prompt = conversation.get_prompt()
            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(self.device)
            stop_string = (
                conversation.sep
                if conversation.sep_style != SeparatorStyle.TWO
                else conversation.sep2
            )
            stopping = []
            if stop_string:
                stopping = [KeywordsStoppingCriteria([stop_string], self.tokenizer, input_ids)]
            output_ids = self.model.generate(
                input_ids=input_ids,
                images=features[None, ...],
                do_sample=False,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=stopping or None,
            )
            output = self.tokenizer.batch_decode(
                output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
            )[0].strip()
            if stop_string and output.endswith(stop_string):
                output = output[:-len(stop_string)].strip()
            conversation.messages[-1][-1] = output
        history.append({"role": "assistant", "content": output})
        return history, conversation, "推理完成 · 整场景特征"


class DemoController:
    def __init__(self, repository: NuScenesSceneRepository, inference: OptionalInferenceEngine) -> None:
        self.repository = repository
        self.inference = inference

    def render(self, scene_token, frame_index, camera, show_boxes, show_tracks):
        try:
            frame = self.repository.get_frame(scene_token, int(frame_index))
            figure_3d, figure_bev = make_plotly_figures(
                frame, show_boxes=show_boxes, show_tracks=show_tracks
            )
            camera_image = self.repository.camera_image(frame, camera)
            model_ready, model_status = self.inference.scene_status(frame.scene)
            scene_id = frame.scene.scene_id or "未映射"
            warning_text = "<br>".join(frame.warnings[:3]) if frame.warnings else "无"
            summary = (
                f"**SCENE** `{scene_id}`　 **DATASET FRAME** `{frame.frame_index:03d}`　"
                f"**POSITION** `{frame.frame_index + 1:02d}/{len(frame.scene.sample_tokens):02d}`　"
                f"**POINTS** `{len(frame.points):,}`  \n"
                f"**SAMPLE** `{frame.sample_token}`　 **BOXES** `{len(frame.boxes)}`　"
                f"**TRACKS** `{len(frame.tracks)}`"
            )
            state = "ONLINE" if model_ready else "VIEW"
            status = f"**{state}** · {model_status}  \n数据告警：{warning_text}"
            return figure_3d, figure_bev, camera_image, summary, status
        except Exception as exc:
            message = f"帧加载失败：{exc}"
            placeholder = empty_plotly_figure(message)
            return placeholder, placeholder, None, "**FRAME ERROR**", message


def create_demo(
    repository: NuScenesSceneRepository,
    inference: OptionalInferenceEngine,
    evaluation: Optional[EvaluationRepository] = None,
):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("缺少 Gradio，请安装 mllm/requirements-demo.txt") from exc

    controller = DemoController(repository, inference)
    initial_scene = repository.scenes[0]
    initial_camera = CAMERA_VIEWS[0]
    initial_render = controller.render(initial_scene.scene_token, 0, initial_camera, True, True)
    initial_chat_ready, initial_chat_status = inference.scene_status(initial_scene)
    scene_choices = [(scene.label, scene.scene_token) for scene in repository.scenes]

    with gr.Blocks(title="B4DL · 4D LiDAR Explorer") as demo:
        gr.HTML(HERO_HTML)
        playing_state = gr.State(False)
        frame_count_state = gr.State(len(initial_scene.sample_tokens))
        conversation_state = gr.State(None)
        timer = gr.Timer(value=0.5, active=False)

        if evaluation is not None and evaluation.enabled:
            effect_page_state = gr.State(0)
            effect_samples, effect_total, effect_page = evaluation.page(page=0)

            def effect_rows(samples):
                return [
                    [
                        sample.sample_id,
                        TASK_LABELS[sample.task],
                        sample.scene_id or "未关联",
                        sample.score_label,
                        sample.status,
                        sample.question[:120],
                    ]
                    for sample in samples
                ]

            def effect_choices(samples):
                return [
                    (
                        f"{sample.score_label:>6} · {TASK_LABELS[sample.task]} · "
                        f"{sample.scene_id or '未关联'} · {sample.question[:72]}",
                        sample.sample_id,
                    )
                    for sample in samples
                ]

            def resolve_effect_sample(sample_id, requested_frame, camera, boxes, tracks):
                if not sample_id:
                    placeholder = empty_plotly_figure("没有可显示的评测样本")
                    return (
                        1, 0, (placeholder, placeholder, None, "**NO SAMPLE**", "无评测样本"),
                        empty_effect_figure("没有可显示的评测样本", "TIMELINE"),
                        None,
                    )
                sample = evaluation.get(sample_id)
                if not sample.scene_id:
                    placeholder = empty_plotly_figure("该旧结果未关联 scene_id")
                    return (
                        1, 0,
                        (placeholder, placeholder, None, "**SCENE UNLINKED**", "请提供 --test_data 关联旧结果"),
                        timeline_figure(sample, 1, 0), sample,
                    )
                try:
                    scene = repository.get_scene_by_id(sample.scene_id)
                    frame_count = len(scene.sample_tokens)
                    frame_index = sample.default_frame if requested_frame is None else int(requested_frame)
                    frame_index = max(0, min(frame_count - 1, frame_index))
                    rendered = controller.render(
                        scene.scene_token, frame_index, camera, boxes, tracks
                    )
                    timeline = timeline_figure(sample, frame_count, frame_index)
                    return frame_count, frame_index, rendered, timeline, sample
                except Exception as exc:
                    placeholder = empty_plotly_figure(str(exc))
                    return (
                        1, 0,
                        (placeholder, placeholder, None, "**SCENE ERROR**", str(exc)),
                        timeline_figure(sample, 1, 0), sample,
                    )

            initial_effect_id = effect_samples[0].sample_id if effect_samples else None
            initial_effect = resolve_effect_sample(
                initial_effect_id, None, initial_camera, True, True
            )
            effect_frame_count, effect_frame_index, effect_render, effect_timeline, effect_sample = initial_effect
            initial_paper_scene = initial_scene
            if effect_sample and effect_sample.scene_id:
                try:
                    initial_paper_scene = repository.get_scene_by_id(effect_sample.scene_id)
                except KeyError:
                    pass
            initial_paper_frames = ", ".join(
                str(value)
                for value in select_frame_indices(len(initial_paper_scene.sample_tokens))
            )

            with gr.Group(elem_classes=["effect-shell"]):
                gr.HTML('<h2 class="effect-heading">MODEL EFFECTS / 模型效果</h2>')
                gr.HTML(metric_cards_html(evaluation.final_scores, evaluation.warnings))
                with gr.Tabs():
                    with gr.Tab("效果总览 / OVERVIEW"):
                        with gr.Row():
                            gr.Plot(
                                overview_metrics_figure(evaluation.final_scores),
                                show_label=False,
                            )
                            gr.Plot(
                                per_task_metrics_figure(evaluation.per_task_metrics),
                                show_label=False,
                            )
                        with gr.Row():
                            gr.Plot(
                                confusion_matrix_figure(evaluation.samples, "existence"),
                                show_label=False,
                            )
                            gr.Plot(
                                confusion_matrix_figure(evaluation.samples, "binary_qa"),
                                show_label=False,
                            )
                        gr.Plot(
                            time_grounding_diagnostics_figure(evaluation.samples),
                            show_label=False,
                        )

                    with gr.Tab("样本诊断 / SAMPLE LAB"):
                        with gr.Row():
                            effect_task = gr.Dropdown(
                                choices=[("全部任务", "all")] + [
                                    (TASK_LABELS[task], task) for task in TASKS
                                ],
                                value="all", label="任务",
                            )
                            effect_status = gr.Dropdown(
                                choices=[
                                    ("全部状态", "all"), ("正确", "correct"),
                                    ("错误", "error"), ("完全匹配", "exact"),
                                    ("部分重叠", "overlap"), ("零重叠", "miss"),
                                    ("无法解析", "unparseable"), ("文本较强", "strong"),
                                    ("文本部分匹配", "partial"), ("文本较弱", "weak"),
                                ],
                                value="all", label="结果状态",
                            )
                            effect_sort = gr.Dropdown(
                                choices=[
                                    ("低分优先", "score_asc"),
                                    ("高分优先", "score_desc"),
                                    ("原始顺序", "source"),
                                ],
                                value="score_asc", label="排序",
                            )
                            effect_query = gr.Textbox(
                                label="搜索", placeholder="问题 / 答案 / scene_id",
                            )
                        with gr.Row():
                            effect_prev_page = gr.Button("◀ 上一页", variant="secondary")
                            effect_page_info = gr.Markdown(
                                f"第 {effect_page + 1} 页 · 当前 {len(effect_samples)} / 共 {effect_total} 条"
                            )
                            effect_next_page = gr.Button("下一页 ▶", variant="secondary")
                        effect_table = gr.Dataframe(
                            headers=["sample_id", "任务", "scene_id", "得分", "状态", "问题"],
                            value=effect_rows(effect_samples),
                            interactive=False,
                            wrap=True,
                            label="筛选结果（每页 50 条）",
                        )
                        effect_sample_picker = gr.Dropdown(
                            choices=effect_choices(effect_samples),
                            value=initial_effect_id,
                            label="选择诊断样本",
                            filterable=True,
                        )

                        with gr.Row(equal_height=False):
                            with gr.Column(scale=7, min_width=600):
                                with gr.Tabs():
                                    with gr.Tab("3D POINT CLOUD"):
                                        effect_plot_3d = gr.Plot(effect_render[0], show_label=False)
                                    with gr.Tab("BEV / 俯视"):
                                        effect_plot_bev = gr.Plot(effect_render[1], show_label=False)
                                    with gr.Tab("CAMERA / 相机"):
                                        effect_camera_image = gr.Image(
                                            effect_render[2], type="pil", interactive=False,
                                            label=initial_camera,
                                        )
                                effect_frame_summary = gr.Markdown(effect_render[3])
                                effect_system_status = gr.Markdown(effect_render[4])
                                effect_frame_slider = gr.Slider(
                                    minimum=0,
                                    maximum=max(0, effect_frame_count - 1),
                                    value=effect_frame_index,
                                    step=1,
                                    label="数据集帧号（0 基）",
                                )
                                effect_timeline_plot = gr.Plot(effect_timeline, show_label=False)
                            with gr.Column(scale=5, min_width=360, elem_id="effect-sample-panel"):
                                effect_diagnosis = gr.HTML(
                                    sample_diagnosis_html(effect_sample) if effect_sample else ""
                                )
                                effect_question = gr.Textbox(
                                    value=effect_sample.question if effect_sample else "",
                                    label="QUESTION", lines=4, interactive=False,
                                )
                                effect_ground_truth = gr.Textbox(
                                    value=effect_sample.ground_truth if effect_sample else "",
                                    label="GROUND TRUTH", lines=6, interactive=False,
                                )
                                effect_prediction = gr.Textbox(
                                    value=effect_sample.prediction if effect_sample else "",
                                    label="PREDICTION", lines=6, interactive=False,
                                )
                                effect_camera = gr.Dropdown(
                                    choices=list(CAMERA_VIEWS), value=initial_camera,
                                    label="相机视角",
                                )
                                with gr.Row():
                                    effect_boxes = gr.Checkbox(value=True, label="真值 3D 框")
                                    effect_tracks = gr.Checkbox(value=True, label="历史轨迹")

                    with gr.Tab("论文案例图 / PAPER CASE"):
                        gr.Markdown(
                            "将评测样本组织为 **5 帧 × 前视 / 后视 / LiDAR BEV** 的论文定性案例图。"
                            "文本高亮项用逗号分隔；留空帧号时自动均匀采样。"
                        )
                        paper_sample_picker = gr.Dropdown(
                            choices=effect_choices(effect_samples),
                            value=initial_effect_id,
                            label="从当前评测页载入样本",
                            filterable=True,
                        )
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=4, min_width=360, elem_id="paper-case-builder"):
                                paper_scene = gr.Dropdown(
                                    choices=scene_choices,
                                    value=initial_paper_scene.scene_token,
                                    label="nuScenes 场景",
                                    filterable=True,
                                )
                                paper_frames = gr.Textbox(
                                    value=initial_paper_frames,
                                    label="帧号（2–8 帧，推荐 5 帧）",
                                    placeholder="例如：0, 10, 20, 30, 39；留空自动选择",
                                )
                                paper_title = gr.Textbox(
                                    value="QUALITATIVE CASE STUDY", label="图标题",
                                )
                                paper_question = gr.Textbox(
                                    value=effect_sample.question if effect_sample else "",
                                    label="QUESTION", lines=3,
                                )
                                with gr.Row():
                                    paper_baseline_label = gr.Textbox(
                                        value="Ground truth", label="左侧模型名",
                                    )
                                    paper_b4dl_label = gr.Textbox(
                                        value="B4DL model (Ours)", label="右侧模型名",
                                    )
                                paper_baseline_answer = gr.Textbox(
                                    value=effect_sample.ground_truth if effect_sample else "",
                                    label="左侧答案 / BASELINE", lines=4,
                                )
                                paper_baseline_highlights = gr.Textbox(
                                    label="左侧黄色高亮短语",
                                    placeholder="vehicles in front, turning left",
                                )
                                paper_b4dl_answer = gr.Textbox(
                                    value=effect_sample.prediction if effect_sample else "",
                                    label="右侧答案 / B4DL", lines=4,
                                )
                                paper_b4dl_highlights = gr.Textbox(
                                    label="右侧绿色高亮短语",
                                    placeholder="vehicles in the back view",
                                )
                                with gr.Row():
                                    paper_boxes = gr.Checkbox(value=True, label="投影真值框")
                                    paper_tracks = gr.Checkbox(value=True, label="LiDAR 历史轨迹")
                                paper_build = gr.Button("生成论文案例图 / EXPORT", variant="primary")
                                paper_status = gr.Markdown("等待生成 · 输出 PNG + PDF")
                                paper_files = gr.File(
                                    label="下载论文图", file_count="multiple", interactive=False,
                                )
                            with gr.Column(scale=8, min_width=640):
                                paper_preview = gr.Image(
                                    type="pil", interactive=False, label="PAPER FIGURE PREVIEW",
                                    elem_id="paper-case-preview",
                                )

            def update_effect_page(task, status, query, sort_order, requested_page):
                page_samples, total, actual_page = evaluation.page(
                    task=task, status=status, query=query, sort_order=sort_order,
                    page=int(requested_page or 0), page_size=50,
                )
                choices = effect_choices(page_samples)
                selected = choices[0][1] if choices else None
                return (
                    effect_rows(page_samples),
                    gr.update(choices=choices, value=selected),
                    f"第 {actual_page + 1} 页 · 当前 {len(page_samples)} / 共 {total} 条",
                    actual_page,
                    gr.update(choices=choices, value=selected),
                )

            effect_page_outputs = [
                effect_table, effect_sample_picker, effect_page_info, effect_page_state,
                paper_sample_picker,
            ]
            for component in (effect_task, effect_status, effect_query, effect_sort):
                component.change(
                    lambda task, status, query, sort_order: update_effect_page(
                        task, status, query, sort_order, 0
                    ),
                    [effect_task, effect_status, effect_query, effect_sort],
                    effect_page_outputs,
                )
            effect_prev_page.click(
                lambda task, status, query, sort_order, page: update_effect_page(
                    task, status, query, sort_order, int(page or 0) - 1
                ),
                [effect_task, effect_status, effect_query, effect_sort, effect_page_state],
                effect_page_outputs,
            )
            effect_next_page.click(
                lambda task, status, query, sort_order, page: update_effect_page(
                    task, status, query, sort_order, int(page or 0) + 1
                ),
                [effect_task, effect_status, effect_query, effect_sort, effect_page_state],
                effect_page_outputs,
            )

            def select_effect_sample(sample_id, camera, boxes, tracks):
                frame_count, frame_index, rendered, timeline, sample = resolve_effect_sample(
                    sample_id, None, camera, boxes, tracks
                )
                return (
                    gr.update(value=frame_index, maximum=max(0, frame_count - 1)),
                    *rendered,
                    sample.question if sample else "",
                    sample.ground_truth if sample else "",
                    sample.prediction if sample else "",
                    sample_diagnosis_html(sample) if sample else "",
                    timeline,
                )

            effect_sample_outputs = [
                effect_frame_slider, effect_plot_3d, effect_plot_bev, effect_camera_image,
                effect_frame_summary, effect_system_status, effect_question,
                effect_ground_truth, effect_prediction, effect_diagnosis,
                effect_timeline_plot,
            ]
            effect_sample_picker.change(
                select_effect_sample,
                [effect_sample_picker, effect_camera, effect_boxes, effect_tracks],
                effect_sample_outputs,
            )

            def change_effect_frame(sample_id, frame_index, camera, boxes, tracks):
                _, _, rendered, timeline, _ = resolve_effect_sample(
                    sample_id, frame_index, camera, boxes, tracks
                )
                return (*rendered, timeline)

            effect_frame_inputs = [
                effect_sample_picker, effect_frame_slider, effect_camera,
                effect_boxes, effect_tracks,
            ]
            effect_frame_outputs = [
                effect_plot_3d, effect_plot_bev, effect_camera_image,
                effect_frame_summary, effect_system_status, effect_timeline_plot,
            ]
            for component in (
                effect_frame_slider, effect_camera, effect_boxes, effect_tracks,
            ):
                component.change(
                    change_effect_frame, effect_frame_inputs, effect_frame_outputs
                )

            def load_paper_sample(sample_id):
                if not sample_id:
                    return (
                        gr.update(), "", "", "", "",
                        "未选择评测样本；可以手动填写场景与文案。",
                    )
                sample = evaluation.get(sample_id)
                scene = initial_scene
                warning = ""
                if sample.scene_id:
                    try:
                        scene = repository.get_scene_by_id(sample.scene_id)
                    except KeyError:
                        warning = f"scene_id {sample.scene_id} 未映射，已保留默认场景。"
                else:
                    warning = "该旧版评测样本没有 scene_id；请手动选择场景。"

                candidates = [
                    int(value) for value in (sample.feat_indices or ())
                    if 0 <= int(value) < len(scene.sample_tokens)
                ]
                if len(set(candidates)) >= 2:
                    candidates = sorted(set(candidates))
                    positions = np.linspace(0, len(candidates) - 1, min(5, len(candidates)))
                    selected = tuple(candidates[int(round(position))] for position in positions)
                else:
                    selected = select_frame_indices(len(scene.sample_tokens))
                frame_text = ", ".join(str(value) for value in selected)
                status = (
                    f"已载入 `{sample.sample_id}` · {TASK_LABELS[sample.task]}"
                    + (f"  \n{warning}" if warning else "")
                )
                return (
                    gr.update(value=scene.scene_token), frame_text, sample.question,
                    sample.ground_truth, sample.prediction, status,
                )

            paper_sample_picker.change(
                load_paper_sample,
                paper_sample_picker,
                [
                    paper_scene, paper_frames, paper_question,
                    paper_baseline_answer, paper_b4dl_answer, paper_status,
                ],
            )

            def export_paper_case(
                scene_token, frames, title, question,
                baseline_label, baseline_answer, baseline_highlights,
                b4dl_label, b4dl_answer, b4dl_highlights,
                boxes, tracks,
            ):
                try:
                    scene = repository.get_scene(scene_token)
                    selected = parse_frame_indices(
                        frames, len(scene.sample_tokens), count=5
                    )
                    artifact = build_paper_case(
                        repository=repository,
                        scene_token=scene_token,
                        frame_indices=selected,
                        question=question,
                        baseline_answer=baseline_answer,
                        b4dl_answer=b4dl_answer,
                        baseline_label=baseline_label or "Baseline",
                        b4dl_label=b4dl_label or "B4DL model (Ours)",
                        baseline_highlights=split_highlights(baseline_highlights),
                        b4dl_highlights=split_highlights(b4dl_highlights),
                        title=title or "QUALITATIVE CASE STUDY",
                        show_boxes=bool(boxes),
                        show_tracks=bool(tracks),
                    )
                    selected_text = ", ".join(str(value) for value in artifact.frame_indices)
                    return (
                        artifact.image,
                        [artifact.png_path, artifact.pdf_path],
                        f"**导出完成** · 帧 `{selected_text}` · PNG / PDF 均已生成",
                    )
                except Exception as exc:
                    return None, None, f"**生成失败** · {exc}"

            paper_build.click(
                export_paper_case,
                [
                    paper_scene, paper_frames, paper_title, paper_question,
                    paper_baseline_label, paper_baseline_answer,
                    paper_baseline_highlights, paper_b4dl_label,
                    paper_b4dl_answer, paper_b4dl_highlights,
                    paper_boxes, paper_tracks,
                ],
                [paper_preview, paper_files, paper_status],
            )

        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=260, elem_id="b4-controls"):
                gr.Markdown("### 场景控制 / CONTROL")
                scene_select = gr.Dropdown(
                    choices=scene_choices, value=initial_scene.scene_token,
                    label="nuScenes 场景", filterable=True,
                )
                camera_select = gr.Dropdown(
                    choices=list(CAMERA_VIEWS), value=initial_camera, label="相机视角",
                )
                frame_slider = gr.Slider(
                    minimum=0, maximum=max(0, len(initial_scene.sample_tokens) - 1),
                    value=0, step=1, label="时间轴 / FRAME INDEX",
                )
                with gr.Row():
                    previous_button = gr.Button("◀", variant="secondary")
                    play_button = gr.Button("▶ 播放", variant="primary")
                    next_button = gr.Button("▶", variant="secondary")
                playback_speed = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.5, step=0.1, label="帧间隔 / 秒",
                )
                show_boxes = gr.Checkbox(value=True, label="显示真值 3D 框")
                show_tracks = gr.Checkbox(value=True, label="显示历史轨迹")
                frame_summary = gr.Markdown(initial_render[3], elem_id="b4-frame-summary")
                system_status = gr.Markdown(initial_render[4], elem_id="b4-status")

            with gr.Column(scale=8, min_width=620, elem_id="b4-viewer"):
                with gr.Tabs():
                    with gr.Tab("3D POINT CLOUD"):
                        plot_3d = gr.Plot(initial_render[0], show_label=False)
                    with gr.Tab("BEV / 俯视"):
                        plot_bev = gr.Plot(initial_render[1], show_label=False)
                    with gr.Tab("CAMERA / 相机"):
                        camera_image = gr.Image(
                            initial_render[2], label=initial_camera, type="pil", interactive=False,
                        )

            with gr.Column(scale=4, min_width=340, elem_id="b4-chat"):
                gr.Markdown("### B4DL 问答 / SCENE CHAT")
                chatbot = gr.Chatbot(value=[], height=540, label="整场景推理")
                chat_input = gr.Textbox(
                    label="问题",
                    placeholder=("询问这个 LiDAR 场景……" if initial_chat_ready else initial_chat_status),
                    interactive=initial_chat_ready,
                    lines=3,
                )
                with gr.Row():
                    send_button = gr.Button(
                        "发送 / RUN", variant="primary", interactive=initial_chat_ready
                    )
                    clear_chat = gr.Button("清空", variant="secondary")

        render_outputs = [plot_3d, plot_bev, camera_image, frame_summary, system_status]
        render_inputs = [scene_select, frame_slider, camera_select, show_boxes, show_tracks]

        def render_callback(scene_token, frame_index, camera, boxes, tracks):
            return controller.render(scene_token, frame_index, camera, boxes, tracks)

        frame_slider.change(render_callback, render_inputs, render_outputs)
        camera_select.change(render_callback, render_inputs, render_outputs)
        show_boxes.change(render_callback, render_inputs, render_outputs)
        show_tracks.change(render_callback, render_inputs, render_outputs)

        def change_scene(scene_token, camera, boxes, tracks):
            scene = repository.get_scene(scene_token)
            rendered = controller.render(scene_token, 0, camera, boxes, tracks)
            ready, status = inference.scene_status(scene)
            return (
                gr.update(value=0, maximum=max(0, len(scene.sample_tokens) - 1)),
                len(scene.sample_tokens), *rendered, [], None, False,
                gr.update(active=False), "▶ 播放",
                gr.update(
                    interactive=ready, value="",
                    placeholder=("询问这个 LiDAR 场景……" if ready else status),
                ),
                gr.update(interactive=ready),
            )

        scene_select.change(
            change_scene,
            [scene_select, camera_select, show_boxes, show_tracks],
            [
                frame_slider, frame_count_state, *render_outputs, chatbot,
                conversation_state, playing_state, timer, play_button, chat_input,
                send_button,
            ],
        )

        def move_frame(delta, scene_token, index, count, camera, boxes, tracks):
            new_index = step_frame(index, count, delta, loop=False)
            rendered = controller.render(scene_token, new_index, camera, boxes, tracks)
            return new_index, *rendered

        previous_button.click(
            lambda scene, index, count, camera, boxes, tracks: move_frame(
                -1, scene, index, count, camera, boxes, tracks
            ),
            [scene_select, frame_slider, frame_count_state, camera_select, show_boxes, show_tracks],
            [frame_slider, *render_outputs],
        )
        next_button.click(
            lambda scene, index, count, camera, boxes, tracks: move_frame(
                1, scene, index, count, camera, boxes, tracks
            ),
            [scene_select, frame_slider, frame_count_state, camera_select, show_boxes, show_tracks],
            [frame_slider, *render_outputs],
        )

        def toggle_play(playing, interval):
            playing = not bool(playing)
            return playing, gr.update(active=playing, value=float(interval)), (
                "■ 停止" if playing else "▶ 播放"
            )

        play_button.click(
            toggle_play, [playing_state, playback_speed], [playing_state, timer, play_button]
        )
        playback_speed.change(
            lambda interval, playing: gr.update(value=float(interval), active=bool(playing)),
            [playback_speed, playing_state], timer,
        )

        def timer_tick(scene_token, index, count, camera, boxes, tracks):
            new_index, keep_playing = advance_playback(index, count)
            if not keep_playing:
                rendered = controller.render(scene_token, index, camera, boxes, tracks)
                return index, *rendered, False, gr.update(active=False), "▶ 播放"
            rendered = controller.render(scene_token, new_index, camera, boxes, tracks)
            return new_index, *rendered, True, gr.update(active=True), "■ 停止"

        timer.tick(
            timer_tick,
            [scene_select, frame_slider, frame_count_state, camera_select, show_boxes, show_tracks],
            [frame_slider, *render_outputs, playing_state, timer, play_button],
        )

        def submit_question(message, history, conversation, scene_token):
            scene = repository.get_scene(scene_token)
            try:
                new_history, new_conversation, status = inference.answer(
                    scene, message, history, conversation
                )
            except Exception as exc:
                new_history = list(history or [])
                if message and message.strip():
                    new_history.append({"role": "user", "content": message.strip()})
                new_history.append({"role": "assistant", "content": f"推理失败：{exc}"})
                new_conversation = conversation
                status = f"推理失败：{exc}"
            return "", new_history, new_conversation, status

        send_button.click(
            submit_question,
            [chat_input, chatbot, conversation_state, scene_select],
            [chat_input, chatbot, conversation_state, system_status],
        )
        chat_input.submit(
            submit_question,
            [chat_input, chatbot, conversation_state, scene_select],
            [chat_input, chatbot, conversation_state, system_status],
        )
        clear_chat.click(lambda: ([], None), None, [chatbot, conversation_state])

    return demo


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B4DL 4D LiDAR Gradio viewer")
    parser.add_argument(
        "--nuscenes_root", "--nuscenes-root", default=os.environ.get("B4DL_NUSCENES_ROOT"),
        help="nuScenes 数据根目录；也可设置 B4DL_NUSCENES_ROOT",
    )
    parser.add_argument("--nuscenes_version", "--nuscenes-version", default="v1.0-trainval")
    parser.add_argument("--scene_metadata", "--scene-metadata", default=None)
    parser.add_argument("--max_points", "--max-points", type=int, default=30_000)
    parser.add_argument("--gpu_id", "--gpu-id", type=int, default=0)
    parser.add_argument("--model_base", "--model-base", default=None)
    parser.add_argument(
        "--pretrain_mm_mlp_adapter", "--pretrain-mm-mlp-adapter", default=None
    )
    parser.add_argument("--stage2", default=None)
    parser.add_argument("--stage3", default=None)
    parser.add_argument("--feat_folder", "--feat-folder", default=None)
    parser.add_argument(
        "--predictions", default=None,
        help="test_b4dl.py 输出的 predictions.json；启用模型效果看板",
    )
    parser.add_argument(
        "--metrics", default=None,
        help="test_b4dl.py 输出的 metrics.json；用于汇总指标图",
    )
    parser.add_argument(
        "--test_data", "--test-data", default=None,
        help="原始 test_qa.json；用于把旧版预测安全关联回 scene_id",
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server_name", "--server-name", default="127.0.0.1")
    parser.add_argument("--server_port", "--server-port", type=int, default=7860)
    args = parser.parse_args(argv)
    if not args.nuscenes_root:
        parser.error("必须提供 --nuscenes_root 或设置 B4DL_NUSCENES_ROOT")
    _, missing = _model_option_state(args)
    if missing:
        parser.error("模型模式参数不完整，缺少：" + ", ".join(missing))
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    repository = NuScenesSceneRepository(
        dataroot=args.nuscenes_root,
        version=args.nuscenes_version,
        scene_metadata=args.scene_metadata,
        max_points=args.max_points,
    )
    evaluation = None
    if args.predictions or args.metrics:
        evaluation = EvaluationRepository.from_files(
            predictions_path=args.predictions,
            metrics_path=args.metrics,
            test_data_path=args.test_data,
        )
    inference = OptionalInferenceEngine(args)
    demo = create_demo(repository, inference, evaluation)
    demo.queue(default_concurrency_limit=4).launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
