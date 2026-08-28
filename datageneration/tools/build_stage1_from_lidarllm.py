#!/usr/bin/env python3
"""
按官方仓库逻辑生成 Stage 1 训练数据（对齐论文 162K）。

官方逻辑（build_stage1_from_lidarllm.py）:
    - 直接读 HF Senqiao/LiDAR-LLM-Nu-Caption 全量数据（train.json, 161,845 条）
    - 按 sample_token → scene 过滤（官方用 assets/sample_token_to_scene.json +
      assets/train_scene_tokens.json 的 699 个训练 scene；本地用
      scene_metadata.json 的 700 个 train scenes 近似，实测 0 泄漏）
    - 不做任何帧映射，conversation id (scene_id) 就是 sample_token 本身

输出格式 (与 mllm/vtimellm/train/dataset.py 兼容):
    {"scene_id": sample_token, "scene_token": scene_token,
     "conversations": [{"from": "human", "value": "<video>\\n{question}"},
                       {"from": "gpt", "value": "{answer_lidar}"}]}

注意: dataset.py 用 source['scene_id'] 拼特征路径 {feat_folder}/{scene_id}.npy,
所以特征文件必须命名为 {sample_token}.npy（见
encoders/lidarclip/extract_pc_features_sample_token.py）。

用法:
    python build_stage1_from_lidarllm.py \
        --llm_train /path/to/lidarllm_train.json \
        --sample_json /path/to/nuScenes/v1.0-trainval/sample.json \
        --scene_metadata /path/to/scene_metadata.json \
        --output /path/to/stage1_train.json
"""
import argparse
import json


def load_sample_to_scene(sample_json_path: str) -> dict:
    """sample.json → {sample_token: scene_token}"""
    with open(sample_json_path) as f:
        samples = json.load(f)
    return {s['token']: s['scene_token'] for s in samples}


def load_train_scene_tokens(scene_metadata_path: str) -> set:
    """scene_metadata.json → 训练 scene_token 集合 (split == 'train')"""
    with open(scene_metadata_path) as f:
        scenes = json.load(f)
    return {s['scene_token'] for s in scenes if s.get('split') == 'train'}


def build(llm_train_path: str, sample_to_scene: dict, train_scene_tokens: set) -> tuple:
    """按官方逻辑过滤并输出，返回 (输出列表, 统计)。"""
    with open(llm_train_path) as f:
        items = json.load(f)

    out = []
    stats = {'total': len(items), 'kept': 0,
             'no_sample': 0, 'not_train_scene': 0}
    for it in items:
        sample_token = it['sample_token']
        scene_token = sample_to_scene.get(sample_token)
        if scene_token is None:
            stats['no_sample'] += 1
            continue
        if scene_token not in train_scene_tokens:
            stats['not_train_scene'] += 1
            continue

        out.append({
            "scene_id": sample_token,
            "scene_token": scene_token,
            "conversations": [
                {"from": "human", "value": f"<video>\n{it['question']}"},
                {"from": "gpt", "value": it.get('answer_lidar') or it.get('answer', '')},
            ],
        })
        stats['kept'] += 1
    return out, stats


def main():
    parser = argparse.ArgumentParser(description="按官方逻辑生成 Stage 1 训练数据")
    parser.add_argument("--llm_train", required=True,
                        help="LiDAR-LLM-Nu-Caption train.json 路径")
    parser.add_argument("--sample_json", required=True,
                        help="nuScenes v1.0-trainval/sample.json 路径")
    parser.add_argument("--scene_metadata", required=True,
                        help="scene_metadata.json 路径 (含 split 划分)")
    parser.add_argument("--output", required=True,
                        help="输出 stage1_train.json 路径")
    args = parser.parse_args()

    print("加载 sample_token → scene_token ...")
    sample_to_scene = load_sample_to_scene(args.sample_json)
    print(f"  {len(sample_to_scene)} 个 sample")

    print("加载训练 scene_token 集合 ...")
    train_scene_tokens = load_train_scene_tokens(args.scene_metadata)
    print(f"  {len(train_scene_tokens)} 个训练 scene")

    print("过滤并输出 ...")
    out, stats = build(args.llm_train, sample_to_scene, train_scene_tokens)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  统计: {stats}")
    print(f"  写入 {len(out)} 条 → {args.output}")


if __name__ == "__main__":
    main()
