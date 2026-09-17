"""Single-frame public inference API and viewer adapter."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch

from vtimellm.constants import IMAGE_TOKEN_INDEX
from vtimellm.conversation import conv_templates
from vtimellm.lidar_visualizer import (
    FrameData,
    SegmentationRender,
    project_full_mask_to_display,
)
from vtimellm.mm_utils import tokenizer_image_token

from .loader import load_reasonseg_model


MASK_COLORS = (
    "#ff4d6d",
    "#00e5ff",
    "#ffb000",
    "#b7ff4a",
    "#b4a7d6",
    "#ff66cc",
    "#66ffcc",
    "#ffd166",
)


@dataclass
class SegmentFrameResult:
    sample_token: str
    text: str
    status: str
    objects: List[SegmentationRender]


class ReasonSegInferenceEngine:
    """Thread-safe model holder implementing ``segment_frame(frame, query)``."""

    def __init__(self, args):
        self.enabled = bool(getattr(args, "seg_checkpoint", None))
        self.device = None
        self.tokenizer = None
        self.model = None
        self._lock = threading.Lock()
        if not self.enabled:
            return
        required = {
            "model_base": getattr(args, "model_base", None),
            "pretrain_mm_mlp_adapter": getattr(args, "pretrain_mm_mlp_adapter", None),
            "stage2": getattr(args, "stage2", None),
            "seg_checkpoint": getattr(args, "seg_checkpoint", None),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("分割模型参数不完整，缺少：" + ", ".join(missing))
        if not torch.cuda.is_available():
            raise RuntimeError("分割模型需要 CUDA；当前可继续使用数据查看模式")
        self.device = torch.device(f"cuda:{args.gpu_id}")
        self.tokenizer, self.model, _ = load_reasonseg_model(
            args,
            b3_checkpoint=args.stage2,
            segmentation_checkpoint=args.seg_checkpoint,
            trainable=False,
        )
        self.model = self.model.to(device=self.device, dtype=torch.float16)
        self.model.eval()

    def status(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "未配置分割 checkpoint"
        return True, "单帧点级分割模型就绪"

    def segment_frame(
        self,
        frame: FrameData,
        query: str,
        *,
        threshold: float = 0.5,
    ) -> SegmentFrameResult:
        if not self.enabled:
            raise RuntimeError("未配置分割 checkpoint")
        if not query or not query.strip():
            raise ValueError("分割问题不能为空")
        if not frame.lidar_path:
            raise ValueError("当前帧没有原始 LiDAR 路径")
        values = np.fromfile(frame.lidar_path, dtype=np.float32)
        if values.size % 5:
            raise RuntimeError(f"无效的 nuScenes 点云文件：{frame.lidar_path}")
        points = torch.from_numpy(values.reshape(-1, 5)[:, :4].copy()).to(
            device=self.device, dtype=torch.float16
        )
        point_batch_indices = torch.zeros(
            points.shape[0], dtype=torch.long, device=self.device
        )
        conversation = conv_templates["v1"].copy()
        conversation.append_message(
            conversation.roles[0],
            "<4DLiDAR>\n<video>\n" + query.strip(),
        )
        conversation.append_message(conversation.roles[1], None)
        input_ids = tokenizer_image_token(
            conversation.get_prompt(),
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)
        attention_mask = torch.ones_like(input_ids)
        with self._lock, torch.inference_mode():
            generated = self.model.generate_and_segment(
                tokenizer=self.tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                points=points,
                point_batch_indices=point_batch_indices,
                threshold=threshold,
                do_sample=False,
                num_beams=1,
                max_new_tokens=160,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        status = generated.status[0]
        suffix = generated.output_ids[0, input_ids.shape[1] :]
        text = self.tokenizer.decode(suffix, skip_special_tokens=False).strip()
        objects: List[SegmentationRender] = []
        if status == "ok":
            masks = generated.mask_probabilities[0]
            class_ids = generated.class_ids[0]
            scores = generated.class_scores[0]
            for index in range(len(masks)):
                class_id = int(class_ids[index])
                class_name = self.model.reasonseg_config.class_names[class_id]
                display_mask = project_full_mask_to_display(
                    frame, masks[index].numpy()
                )
                objects.append(
                    SegmentationRender(
                        label=class_name,
                        color=MASK_COLORS[index % len(MASK_COLORS)],
                        mask=display_mask,
                        score=float(scores[index]),
                    )
                )
        return SegmentFrameResult(
            sample_token=frame.sample_token,
            text=text,
            status=status,
            objects=objects,
        )
