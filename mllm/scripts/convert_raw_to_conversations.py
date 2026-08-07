"""
将 HF 下载的 raw 格式数据（question/answer 字段）转换为训练所需的 conversations 格式。

用法：
    python convert_raw_to_conversations.py

输入文件（raw 格式）：
    - stage2.json           # 68,695 条
    - stage3.json           # 79,576 条

输出文件（conversations 格式）：
    - stage2_conversations.json      # 全量，Stage2 训练用
    - stage3_conversations.json      # 全量
    - stage2_train.json / stage2_val.json / stage2_test.json
    - stage3_train.json / stage3_val.json / stage3_test.json

划分依据：split_scenes.json（train 559 / val 69 / test 71 个 scene）
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
B4DL_DATASET_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "b4dl_dataset")


def load_json(path: str):
    print(f"  读取: {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    print(f"  写入: {path} ({len(data)} 条)")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def convert_raw_to_conversations(raw_items: list) -> list:
    """
    将 raw 格式转为 conversations 格式。

    raw 格式:
        {"split": "train", "scene_token": "...", "scene_id": "...",
         "human_annotation": "...", "question": "...", "answer": "..."}

    conversations 格式:
        {"scene_id": "...", "scene_token": "...", "split": "train",
         "conversations": [
             {"from": "human", "value": "<video>\\n问题"},
             {"from": "gpt",   "value": "答案"}
         ]}
    """
    converted = []
    for item in raw_items:
        question = item["question"].strip()
        answer = item["answer"].strip()

        # 如果 question 已经以 <video> 开头，不再重复添加
        if not question.startswith("<video>"):
            question = "<video>\n" + question

        converted.append({
            "scene_id": item["scene_id"],
            "scene_token": item.get("scene_token", ""),
            "split": item.get("split", "train"),
            "conversations": [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ],
        })
    return converted


def split_by_scene(data: list, split_scenes: dict) -> dict:
    """
    按 scene_id 将数据划分为 train/val/test。
    split_scenes: {"train": [...], "val": [...], "test": [...]}
    """
    train_ids = set(split_scenes["train"])
    val_ids = set(split_scenes["val"])
    test_ids = set(split_scenes["test"])

    splits = {"train": [], "val": [], "test": [], "unknown": []}
    for item in data:
        sid = item["scene_id"]
        if sid in train_ids:
            splits["train"].append(item)
        elif sid in val_ids:
            splits["val"].append(item)
        elif sid in test_ids:
            splits["test"].append(item)
        else:
            splits["unknown"].append(item)

    return splits


def process_stage(raw_path: str, conv_path: str, prefix: str, split_scenes: dict, output_dir: str):
    """处理一个 stage：转换 + 划分"""
    print(f"\n{'='*50}")
    print(f"处理: {os.path.basename(raw_path)}")
    print(f"{'='*50}")

    raw_data = load_json(raw_path)
    print(f"  raw 格式: {len(raw_data)} 条")

    # 统计唯一 scene
    raw_scenes = set(item["scene_id"] for item in raw_data)
    print(f"  唯一 scene: {len(raw_scenes)}")

    # 转换为 conversations 格式
    conv_data = convert_raw_to_conversations(raw_data)
    save_json(conv_data, conv_path)

    # 按 scene 划分
    splits = split_by_scene(conv_data, split_scenes)
    for split_name in ["train", "val", "test"]:
        if splits[split_name]:
            out_path = os.path.join(output_dir, f"{prefix}_{split_name}.json")
            save_json(splits[split_name], out_path)

    if splits["unknown"]:
        print(f"  ⚠ 警告: {len(splits['unknown'])} 条数据的 scene_id 不在 split_scenes 中！")
        unknown_scenes = set(item["scene_id"] for item in splits["unknown"])
        print(f"  未知 scene: {list(unknown_scenes)[:10]}...")

    # 汇总
    total_split = sum(len(splits[k]) for k in ["train", "val", "test"])
    print(f"  划分: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}, 合计={total_split}")


def process_test_files(test_dir: str, output_path: str):
    """处理测试集 6 个 raw JSON，转换为 conversations 格式并合并为一个文件。

    每个测试 JSON（existence.json 等）的 task 由文件名推断。
    输出格式（eval 脚本期望的格式）：
        {"scene_id": "...", "task": "existence", "conversations": [...]}
    """
    TASK_NAMES = [
        "existence", "binary", "time_grounding",
        "description", "temporal_understanding", "comprehensive",
    ]

    print(f"\n{'='*50}")
    print(f"处理测试集: {test_dir}")
    print(f"{'='*50}")

    all_items = []
    for task in TASK_NAMES:
        fpath = os.path.join(test_dir, f"{task}.json")
        if not os.path.exists(fpath):
            print(f"  ⚠ 跳过: {fpath} (文件不存在)")
            continue

        raw_data = load_json(fpath)
        print(f"  {task}.json: {len(raw_data)} 条")

        for item in raw_data:
            question = item["question"].strip()
            answer = item["answer"].strip()
            if not question.startswith("<video>"):
                question = "<video>\n" + question

            all_items.append({
                "scene_id": item["scene_id"],
                "scene_token": item.get("scene_token", ""),
                "split": item.get("split", "test"),
                "task": task,
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer},
                ],
            })

    save_json(all_items, output_path)

    # 统计
    tasks = set(item["task"] for item in all_items)
    for t in sorted(tasks):
        cnt = sum(1 for it in all_items if it["task"] == t)
        print(f"  → {t}: {cnt} 条")
    print(f"  合计: {len(all_items)} 条")


def main():
    print("=" * 60)
    print("B4DL 数据集格式转换: raw → conversations")
    print("=" * 60)

    # 数据目录: mllm/b4dl_dataset/
    data_dir = B4DL_DATASET_DIR

    # 加载划分
    split_path = os.path.join(data_dir, "split_scenes.json")
    if not os.path.exists(split_path):
        print(f"错误: 找不到 {split_path}")
        sys.exit(1)
    split_scenes = load_json(split_path)
    print(f"split_scenes: train={len(split_scenes['train'])}, val={len(split_scenes['val'])}, test={len(split_scenes['test'])}")

    # ── 训练集：stage2 / stage3 ──
    process_stage(
        raw_path=os.path.join(data_dir, "stage2.json"),
        conv_path=os.path.join(data_dir, "stage2_conversations.json"),
        prefix="stage2",
        split_scenes=split_scenes,
        output_dir=data_dir,
    )

    process_stage(
        raw_path=os.path.join(data_dir, "stage3.json"),
        conv_path=os.path.join(data_dir, "stage3_conversations.json"),
        prefix="stage3",
        split_scenes=split_scenes,
        output_dir=data_dir,
    )

    # ── 测试集：6 个任务 raw → conversations 合并 ──
    hf_test_dir = os.path.join(
        os.path.dirname(data_dir),  # mllm/
        "..", "dataset", "nuScenes-B4DL", "dataset", "test"
    )
    hf_test_dir = os.path.normpath(hf_test_dir)

    if os.path.isdir(hf_test_dir):
        process_test_files(
            test_dir=hf_test_dir,
            output_path=os.path.join(data_dir, "test_qa.json"),
        )
    else:
        print(f"\n  ⚠ 未找到 HF 测试集目录: {hf_test_dir}")
        print(f"    跳过测试集转换。如需转换，请手动指定路径。")

    print(f"\n{'='*60}")
    print("全部转换完成!")
    print(f"{'='*60}")
    print(f"\n生成的文件:")
    print(f"  stage2_conversations.json  → Stage2 训练用（全量）")
    print(f"  stage3_conversations.json  → Stage3 全量")
    print(f"  stage2_train/val/test.json → Stage2 划分")
    print(f"  stage3_train/val/test.json → Stage3 划分")
    print(f"  test_qa.json               → 六任务测试集合并（评测用）")


if __name__ == "__main__":
    main()
