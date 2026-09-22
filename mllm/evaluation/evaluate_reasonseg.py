#!/usr/bin/env python3
"""Free-generation evaluation for B4DL reasoning segmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


MLLM_ROOT = Path(__file__).resolve().parents[1]
if str(MLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(MLLM_ROOT))

from vtimellm.constants import IMAGE_TOKEN_INDEX
from vtimellm.conversation import conv_templates
from vtimellm.mm_utils import tokenizer_image_token
from vtimellm.segmentation.data import ReasonSegDataset
from vtimellm.segmentation.loader import load_reasonseg_model
from vtimellm.segmentation.metrics import SegmentationMetricAccumulator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--pretrain-mm-mlp-adapter", required=True)
    parser.add_argument("--b3-checkpoint", required=True)
    parser.add_argument("--seg-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument(
        "--encoder-dtype",
        choices=("same", "fp32"),
        default="fp32",
        help="spconv 点云编码器的权重精度；fp16 会让部分帧的特征变成 NaN，"
             "贪心解码随即锁死在 <unk>（id 0）上，故默认 fp32。",
    )
    args = parser.parse_args()
    eval_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    if not torch.cuda.is_available():
        raise RuntimeError("ReasonSeg evaluation requires CUDA")
    # Adapter expected by the shared B3 loader.
    args.stage2 = args.b3_checkpoint
    device = torch.device(f"cuda:{args.gpu_id}")
    tokenizer, model, _ = load_reasonseg_model(
        args,
        b3_checkpoint=args.b3_checkpoint,
        segmentation_checkpoint=args.seg_checkpoint,
        trainable=False,
    )
    model = model.to(device=device, dtype=eval_dtype).eval()
    point_dtype = eval_dtype
    if args.encoder_dtype == "fp32":
        model.point_encoder.to(torch.float32)
        model.scene_compressor.to(torch.float32)
        point_dtype = torch.float32
        encode_points = model.encode_points

        def encode_points_fp32(points, point_batch_indices):
            encoding, padded, valid, scene, indices = encode_points(
                points.to(torch.float32), point_batch_indices
            )
            return encoding, padded.to(eval_dtype), valid, scene.to(eval_dtype), indices

        model.encode_points = encode_points_fp32
    dataset = ReasonSegDataset(args.manifest, dataroot=args.dataroot,
                               config=model.reasonseg_config)
    output_dir = Path(args.output_dir)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    accumulator = SegmentationMetricAccumulator(model.reasonseg_config.num_classes)
    predictions = []
    sample_count = len(dataset) if not args.max_samples else min(len(dataset), args.max_samples)
    for index in range(sample_count):
        sample = dataset[index]
        input_ids = tokenize_query(tokenizer, sample["query"]).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)
        points = sample["points"].to(device=device, dtype=point_dtype)
        point_batch_indices = torch.zeros(len(points), dtype=torch.long, device=device)
        generated = model.generate_and_segment(
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            points=points,
            point_batch_indices=point_batch_indices,
            threshold=args.threshold,
            do_sample=False,
            num_beams=1,
            max_new_tokens=160,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        predicted_masks = generated.mask_probabilities[0].numpy()
        predicted_classes = generated.class_ids[0].numpy()
        valid_targets = sample["object_valid_mask"].numpy().astype(bool)
        target_masks = sample["target_masks"].numpy()[valid_targets]
        target_classes = sample["target_classes"].numpy()[valid_targets]
        status = generated.status[0]
        accumulator.update(
            predicted_masks,
            predicted_classes,
            target_masks,
            target_classes,
            token_status=status,
        )
        mask_file = masks_dir / f"{index:06d}_{sample['sample_token']}.npz"
        np.savez_compressed(
            mask_file,
            predicted_masks=predicted_masks,
            predicted_classes=predicted_classes,
            target_masks=target_masks,
            target_classes=target_classes,
        )
        suffix = generated.output_ids[0, input_ids.shape[1] :]
        predictions.append(
            {
                "index": index,
                "sample_token": sample["sample_token"],
                "query": sample["query"],
                "answer": sample["answer"],
                "generated": tokenizer.decode(suffix, skip_special_tokens=False).strip(),
                "status": status,
                "mask_file": str(mask_file.relative_to(output_dir)).replace("\\", "/"),
            }
        )
        if (index + 1) % 100 == 0:
            print(f"evaluated {index + 1}/{sample_count}")
    metrics = accumulator.compute()
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def tokenize_query(tokenizer, query: str) -> torch.Tensor:
    conversation = conv_templates["v1"].copy()
    conversation.append_message(
        conversation.roles[0], "<4DLiDAR>\n<video>\n" + query.strip()
    )
    conversation.append_message(conversation.roles[1], None)
    return tokenizer_image_token(
        conversation.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    )


if __name__ == "__main__":
    raise SystemExit(main())
