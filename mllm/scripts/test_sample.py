#!/usr/bin/env python3
"""
B4DL 单条样本推理测试
用于快速验证训练后的模型是否能正常工作。

用法:
  # 随机测试一条
  python scripts/test_sample.py \
      --model_base ./base_model/vicuna-v1-5-7b \
      --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
      --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2 \
      --feat_folder ../encoders/lidarclip/b4dl/stage2_features

  # 测试指定场景
  python scripts/test_sample.py ... --scene_id 004876401

  # 交互模式（输入自定义问题）
  python scripts/test_sample.py ... --interactive
"""
import os
import sys
import json
import argparse
import random

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vtimellm.model.builder import load_pretrained_model
from vtimellm.utils import disable_torch_init
from vtimellm.constants import IMAGE_TOKEN_INDEX
from vtimellm.conversation import conv_templates, SeparatorStyle
from vtimellm.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria


def run_inference(model, features, query, tokenizer, temperature=0.05, max_new_tokens=512):
    """Run single inference with LiDAR features."""
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=features.unsqueeze(0).cuda() if features.dim() == 2 else features[None,].cuda(),
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    return outputs.strip()


def classify(gt):
    """Classify ground truth answer type."""
    import re
    gt_lower = gt.strip().lower().rstrip('.')
    if gt_lower in ('yes', 'no'):
        return 'binary'
    if re.match(r'^from frame \d+ to frame \d+$', gt_lower):
        return 'frame_range'
    return 'categorical'


def main():
    parser = argparse.ArgumentParser(description="B4DL single sample test")
    parser.add_argument("--model_base", type=str, required=True)
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, required=True)
    parser.add_argument("--stage2", type=str, default=None)
    parser.add_argument("--stage3", type=str, default=None)
    parser.add_argument("--feat_folder", type=str,
                        default="../encoders/lidarclip/b4dl/stage2_features")
    parser.add_argument("--data_path", type=str,
                        default="./b4dl_dataset/stage2_val.json")
    parser.add_argument("--scene_id", type=str, default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.05)
    args = parser.parse_args()

    # Load model
    disable_torch_init()
    print("Loading model...")
    tokenizer, model, _ = load_pretrained_model(args, args.stage2, args.stage3)
    model = model.cuda().to(torch.float16)
    tokenizer.pad_token = tokenizer.unk_token
    print("Model loaded.\n")

    # Load data
    data = json.load(open(args.data_path, "r"))

    # Select sample
    sample = None
    if args.scene_id:
        for d in data:
            if d['scene_id'] == args.scene_id:
                sample = d
                break
        if sample is None:
            print(f"[ERROR] scene_id={args.scene_id} not found in {args.data_path}")
            sys.exit(1)
    elif args.index is not None:
        sample = data[args.index]
    else:
        sample = random.choice(data)

    scene_id = sample['scene_id']
    query = sample['conversations'][0]['value']
    gt = sample['conversations'][1]['value']
    answer_type = classify(gt)

    # Load features
    feat_path = os.path.join(args.feat_folder, f"{scene_id}.npy")
    if not os.path.exists(feat_path):
        print(f"[ERROR] Feature not found: {feat_path}")
        sys.exit(1)
    features = torch.from_numpy(np.load(feat_path)).to(torch.float16)
    print(f"Features shape: {features.shape}")

    # Show sample info
    print("=" * 60)
    print(f"Scene ID:   {scene_id}")
    print(f"Query:      {query}")
    print(f"Ground Truth: {gt}")
    print(f"Answer Type:  {answer_type}")
    print(f"Feature shape: {features.shape}")
    print("=" * 60)

    # Run inference
    pred = run_inference(model, features, query, tokenizer, args.temperature)
    correct = pred.strip().lower() == gt.strip().lower()

    print(f"\nPrediction:  {pred}")
    print(f"Correct:     {'✓ YES' if correct else '✗ NO'}")

    # Interactive mode
    if args.interactive:
        print("\n" + "=" * 60)
        print("交互模式（输入 'quit' 退出）")
        print("=" * 60)
        while True:
            try:
                user_q = input("\n> 请输入问题: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_q.lower() in ('quit', 'exit', 'q'):
                break
            if not user_q:
                continue
            full_query = f"<video>\n{user_q}"
            answer = run_inference(model, features, full_query, tokenizer, args.temperature)
            print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()
