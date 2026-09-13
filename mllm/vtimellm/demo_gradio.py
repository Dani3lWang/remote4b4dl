"""B4DL 4D LiDAR scene explorer and optional model chat UI.

Viewer-only mode needs nuScenes plus the demo requirements. Model chat is
enabled only when all model/feature arguments are supplied as a complete set.
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
"""

HERO_HTML = """
<section id="b4-hero">
  <div class="b4-kicker"><span class="b4-live"></span>B4DL / SENSOR OPERATIONS</div>
  <h1 class="b4-title">4D LiDAR 场景查看器</h1>
  <div class="b4-subtitle">NUSCENES TIMELINE · 3D / BEV · GROUND-TRUTH TRACKS · OPTIONAL LLM</div>
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
            frame_indices = torch.arange(features.shape[0], dtype=torch.long)
            output_ids = self.model.generate(
                input_ids=input_ids,
                images=features[None, ...],
                frame_indices=[frame_indices],
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
                f"**SCENE** `{scene_id}`　 **FRAME** `{frame.frame_index + 1:02d}/"
                f"{len(frame.scene.sample_tokens):02d}`　 **POINTS** `{len(frame.points):,}`  \n"
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


def create_demo(repository: NuScenesSceneRepository, inference: OptionalInferenceEngine):
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
    inference = OptionalInferenceEngine(args)
    demo = create_demo(repository, inference)
    demo.queue(default_concurrency_limit=4).launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
