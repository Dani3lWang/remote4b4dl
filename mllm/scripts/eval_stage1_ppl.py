#!/usr/bin/env python3
"""
Stage1 mm_projector 质量评估：计算验证集上的 Perplexity。

Perplexity = exp(cross_entropy_loss)，越低越好。
数值含义：模型对每个 token 的平均"困惑度"（相当于从多少个选项中选一个）。

用法:
    python scripts/eval_stage1_ppl.py \
        --model_base ./base_model/vicuna-v1-5-7b \
        --mm_projector ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
        --data_path ./b4dl_dataset/stage1_val.json \
        --feat_folder ../encoders/lidarclip/b4dl/stage1_features \
        --max_samples 500
"""
import os, sys, json, argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vtimellm.model.builder import load_pretrained_model
from vtimellm.utils import disable_torch_init
from vtimellm.mm_utils import tokenizer_image_token
from vtimellm.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX


def compute_ppl(model, tokenizer, features, input_ids, labels, device):
    """计算单样本 perplexity"""
    features = features.to(device=device, dtype=model.dtype)
    input_ids = input_ids.to(device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            images=features.unsqueeze(0) if features.dim() == 2 else features[None,],
            labels=labels.unsqueeze(0).to(device),
            use_cache=False,
        )
        loss = outputs.loss
        # 如果 loss 是 per-token 的，取 mean
        if loss.dim() > 0:
            loss = loss.mean()
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base", required=True)
    parser.add_argument("--mm_projector", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--feat_folder", required=True)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--old_projector", type=str, default=None,
                        help="旧 mm_projector 路径（可选，用于对比）")
    args = parser.parse_args()

    disable_torch_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    data = json.load(open(args.data_path))
    if args.max_samples and len(data) > args.max_samples:
        import random; random.seed(42)
        data = random.sample(data, args.max_samples)
    print(f"评估样本: {len(data)}")

    # 评估函数
    def evaluate(mm_path, label):
        # 构造临时 args 对象给 load_pretrained_model
        class Args: pass
        model_args = Args()
        model_args.model_base = args.model_base
        model_args.pretrain_mm_mlp_adapter = mm_path
        model_args.stage2 = None
        model_args.stage3 = None

        tokenizer, model, _ = load_pretrained_model(model_args, None, None)
        model = model.to(device).to(torch.float16)
        model.eval()
        tokenizer.pad_token = tokenizer.unk_token

        total_loss, count = 0.0, 0
        for item in tqdm(data, desc=label):
            scene_id = item['scene_id']
            feat_path = os.path.join(args.feat_folder, f"{scene_id}.npy")
            if not os.path.exists(feat_path):
                continue

            features = torch.from_numpy(np.load(feat_path)).to(torch.float16)

            # 构造 prompt：plain 格式
            human_val = item['conversations'][0]['value']  # 已含<video>
            gpt_val = item['conversations'][1]['value']
            prompt = human_val + '\n' + gpt_val  # plain 格式就是拼接

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors='pt')
            # labels: human 部分设为 IGNORE，只对 gpt 部分计算 loss
            # human 的 token 数 = <video> token 被替换为 IMAGE_TOKEN_INDEX
            human_ids = tokenizer_image_token(human_val, tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors='pt')
            labels = input_ids.clone()
            labels[:len(human_ids)] = IGNORE_INDEX

            loss = compute_ppl(model, tokenizer, features, input_ids, labels, device)
            total_loss += loss
            count += 1

        avg_loss = total_loss / count if count > 0 else float('inf')
        ppl = np.exp(avg_loss)
        print(f"\n{label}: samples={count}, avg_loss={avg_loss:.4f}, perplexity={ppl:.1f}")
        return avg_loss, ppl

    # 评估新 projector
    new_loss, new_ppl = evaluate(args.mm_projector, "新 Stage1 (95k)")

    # 评估旧 projector（如果提供）
    if args.old_projector and os.path.exists(args.old_projector):
        old_loss, old_ppl = evaluate(args.old_projector, "旧 Stage1 (699)")
        print(f"\n{'='*50}")
        print(f"对比:")
        print(f"  旧 (699):  loss={old_loss:.4f}, ppl={old_ppl:.1f}")
        print(f"  新 (95k):  loss={new_loss:.4f}, ppl={new_ppl:.1f}")
        if new_ppl < old_ppl:
            print(f"  ✓ 新 projector 更优 (ppl 低 {old_ppl - new_ppl:.1f})")
        else:
            print(f"  ✗ 旧 projector 在 val 上 ppl 更低，但可能过拟合")

    # 输出关键指标
    print(f"\n{'='*50}")
    print(f"Key metric: Perplexity = {new_ppl:.1f}")
    print(f"含义: 模型对每个 token 相当于从 {new_ppl:.0f} 个选项中做选择")


if __name__ == "__main__":
    main()
