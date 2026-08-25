#!/usr/bin/env python3
"""
build_stage2_full_train.py — 按论文官方划分构建 Stage2 全量训练数据
====================================================================
论文 §3.3：B4DL 数据集由 850 个 nuScenes scene 组成，官方划分为
700 train / 150 test，train 共 148,271 条 QA（stage2 68,695 + stage3 79,576）。

HF 发布的 `ccho4702/nuScenes-B4DL` train 目录（stage2.json + stage3.json）
本身就是论文的官方训练集：699 个 train scene 与官方 test_qa.json 的
150 个 scene 零重叠，无需（也不应）再做任何额外划分。

本脚本替代旧的 create_splits.py（80/10/10 自创划分会把 850 个 scene
混在一起切，切出的内部 test 与官方测试集冲突，导致训练数据被砍到
118,722 条/559 scenes）：

1. 读取 HF 原始平铺格式 {question, answer, human_annotation, ...}
2. 转为 mllm 训练所需的 conversations 格式（保留 scene_id 等字段）
3. 输出 stage2_full_train_148k.json（148,271 条，未注入 metatoken）

后续注入 metatoken（per-seq 或 --frame_ctx）由 inject_metatoken.py 完成。

用法：
    python scripts/build_stage2_full_train.py \
        --input_dir ../dataset/nuScenes-B4DL/dataset/train \
        --output_dir ./b4dl_dataset
"""
import os
import json
import re
import argparse

_TG_PATTERN = re.compile(r'from\s+frames?\s+\d+\s+to\s+frames?\s+\d+', re.I)


def to_conversations(items: list, tag_tg: bool = False) -> list:
    out = []
    for it in items:
        q = it.get("question", "").strip()
        a = it.get("answer", "").strip()
        if not q or not a:
            continue
        item_out = {
            "scene_id": it["scene_id"],
            "scene_token": it.get("scene_token"),
            "split": it.get("split", "train"),
            "human_annotation": it.get("human_annotation", ""),
            "conversations": [
                {"from": "human", "value": q},
                {"from": "gpt", "value": a},
            ],
        }
        # 仅 stage2 简单任务（存在性/二分类/time_grounding）打 TG 标签：
        # 答案为 "from frame(s) X to frame(s) Y" 的是 time_grounding
        # （13,124 条，与论文 Table 2 一致）。stage3 复杂任务的 prose 答案
        # 也含该短语但不属于 TG，不在此打标签。标签供 inject_metatoken.py
        # --answer_frames 精确限定 GT 归属回退。
        if tag_tg and _TG_PATTERN.search(a):
            item_out["task"] = "time_grounding"
        out.append(item_out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="HF 原始数据目录 (含 stage2.json / stage3.json)")
    parser.add_argument("--output_dir", default="./b4dl_dataset")
    parser.add_argument("--output_name", default="stage2_full_train_148k.json")
    args = parser.parse_args()

    conv = []
    # 分别转换：只有 stage2 来源打 TG 标签（合并后无法区分来源）
    for name, tag_tg in [("stage2.json", True), ("stage3.json", False)]:
        path = os.path.join(args.input_dir, name)
        items = json.load(open(path))
        print(f"{name}: {len(items)} 条" + (" (打 TG 标签)" if tag_tg else ""))
        conv += to_conversations(items, tag_tg=tag_tg)
        # 同时写出分阶段数据（论文/官方训练法：stage2 简单任务先训，
        # merge 后 stage3 复杂任务再训；合并版仅作对照）
        sep_name = name.replace(".json", "_train.json")
        with open(os.path.join(args.output_dir, sep_name), "w",
                  encoding="utf-8") as f:
            json.dump(to_conversations(items, tag_tg=tag_tg), f,
                      ensure_ascii=False)
        print(f"  → {args.output_dir}/{sep_name}")
    scenes = sorted({it["scene_id"] for it in conv})

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(conv, f, ensure_ascii=False)
    print(f"\n合并完成: {len(conv)} 条 / {len(scenes)} scenes")
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
