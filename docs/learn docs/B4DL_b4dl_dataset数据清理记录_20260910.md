# b4dl_dataset 数据清理记录（2026-09-10）

> 目录：`mllm/b4dl_dataset/`（gitignore，**不在 git 跟踪范围内，删除不可回滚**，故本记录即其台账）
> 结果：**1.5 GiB / 34 个文件 → 501 MiB / 15 个文件**，删除 19 项共 ≈ 1.04 GiB
> 原则：只留「来源数据（官方/评测用）」+「当前 B3 路线所需」；可脚本再生的实验变体一律删除
> 判据：删除前已核对 (a) 无运行中作业引用、(b) 保留集结构完整、(c) 每个删除项有再生命令

## 1. 保留清单（15 项 / 501 MiB）

| 文件 | 体积 | 类别 | 消费者 / 保留理由 |
|---|---|---|---|
| `stage1_train.json` | 122M | 来源数据 | Stage1 训练（161,629 条，162K 官方对齐；`scripts/stage1.sh` 默认） |
| `stage1_val.json` | 28M | 来源数据 | Stage1 验证（42,597 条；`scripts/eval_stage1_ppl.py`） |
| `test_qa.json` | 11M | 评测数据 | **主评测集** 30,145 条 6 任务（B3 全量评测输入，TG 2,783 条） |
| `test_qa_simple_only.json` | 3.9M | 评测数据 | 简单任务子集 14,078 条（binary_qa 7,525 + existence 3,770 + TG 2,783），`eval_results/official_ckpt_nometa` 诊断用 |
| `ego_metadata.json` | 2.4M | 评测元数据 | metatoken 注入（`--ego_meta`，训练+评测双侧） |
| `ego_frame_motion.json` | 9.2M | 评测元数据 | 逐帧运动表（`--frame_motion`，meta2/seqv2/seqv3 注入与评测都必须） |
| `stage2_full_train_148k.json` | 61M | 官方基座 | 官方 148,271 条直转（未注入）；**所有变体的再生基座**，TG 标签 13,124 条完整保留 |
| `stage2_full_train_seqv3_meta2_148k.json` | 131M | **B3 训练数据** | 当前最优基线（mIoU 0.3467）；113,053 条含 `feat_indices`/`feat_range` |
| `stage2_full_train_seqv3_meta2_oversampled_150k.json` | 132M | 临时保留 | B4a 负结果数据（150,222 条）——`mllm/_vram_smoke_bs8/run_measure.sh` **待跑作业的 `--data_path`**，别删；该作业是 bs8 显存测量，跑完即可删（再生见 §3.4） |
| `convert_raw_to_conversations.py` | 8.5K | 脚本 | 原始转换脚本（旧 80/10/10 流程用，保留备查） |
| `stage2.json`、`stage3.json`、`stage2_conversations.json`、`stage3_conversations.json`、`split_scenes.json` | 133B ×4 + 130B | 占位文件 | 上游仓库的 **Git LFS 指针**（内容从未 fetch，不是有效 JSON）；体积可忽略故保留 |
| `.cache/huggingface/` | 60K | HF 缓存标记 | `dataset/` 下载缓存元数据 |

## 2. 删除清单（19 项 ≈ 1.04 GiB）

### 2.1 备份文件（`.bak*`，202M）
| 文件 | 体积 | 说明 |
|---|---|---|
| `stage1_train_700scenes_161845.json.bak` | 122M | 161,845 条未按特征可用性过滤版（现行 `stage1_train.json` 161,629 条即其过滤版） |
| `stage1_train_frameid_95048.json.bak` | 69M | frame_id 键控旧方案（95,048 条），已退役 |
| `ego_metadata.json.bak.per_scene` | 343K | per-scene 改造前备份 |

### 2.2 旧 80/10/10 自创划分链（已废弃，见 CLAUDE.md「数据划分」段）（287M）
| 文件 | 体积 | 条数 | 说明 |
|---|---|---|---|
| `stage2_full_train.json` | 90M | 118,722 | 自创划分把训练集砍到 559 scenes，与官方测试集冲突 |
| `stage2_full_train_seq.json` | 90M | 118,722 | 同上 + seq 注入 |
| `stage2_full_train_seqv2.json` | 95M | 118,722 | 同上 + seqv2 注入 |
| `stage2_val.json` / `stage2_test.json` | 1.7M / 1.8M | 6,723 / 7,071 | 自创划分 val/test（`scripts/verify.sh` 曾引用） |
| `stage3_val.json` / `stage3_test.json` | 3.1M / 3.3M | 7,710 / 8,045 | 同上 |
| `stage2_full_val.json` | 4.8M | 14,433 | 同上 |

⚠️ 该链的再生依赖上游 LFS 的 `stage2_conversations.json` / `stage3_conversations.json`（本地仅指针）+ `create_splits.py`——**不建议再生**，官方划分已取代它。

### 2.3 被取代的 148k 注入代次（B0/B1/B2 线）（650M）
| 文件 | 体积 | 条数 | 说明 |
|---|---|---|---|
| `stage2_full_train_148k_seq.json` | 125M | 148,271 | seq v1 注入（旧渲染，无 `feat_indices`），被 seqv2 取代 |
| `stage2_full_train_seqv2_148k.json` | 131M | 148,271 | seqv2（B0 前身），被 seqv3 取代 |
| `stage2_full_train_seqv3_148k.json` | 132M | 148,271 | seqv3（B0/B1/B2 训练数据；`feat_indices` 113,053），被 meta2 取代 |
| `stage2_train_seqv3.json` | 56M | 68,695 | 论文两阶段 Phase A（简单任务）数据 |
| `stage3_train_seqv3.json` | 76M | 79,576 | 论文两阶段 Phase B（复杂任务）数据 |
| `stage2_train.json` / `stage3_train.json` | 23M / 38M | 68,695 / 79,576 | 官方 stage2/stage3 train 直转（未注入），仅两阶段路径消费 |

### 2.4 其他
| 文件 | 体积 | 说明 |
|---|---|---|
| `__pycache__/convert_raw_to_conversations.cpython-310.pyc` | 7.5K | 顺手清理的缓存 |

> B0/B1/B2/B4a 的**评测产物与 adapter 备份均未受影响**（`mllm/eval_results/` + `backups/checkpoints_*.tar.gz`），历史数值照旧可查；删掉的只是「重训用的数据」，且都有 §3 的再生路径。

## 3. 变体再生配方（全部现成脚本，逐条可复制）

```bash
cd /root/autodl-tmp/wql/mmb4dl/mllm
```

### 3.1 官方 148k 基座（同时产出 `stage2_train.json` / `stage3_train.json`）
```bash
python scripts/build_stage2_full_train.py \
    --input_dir ../dataset/nuScenes-B4DL/dataset/train \
    --output_dir ./b4dl_dataset
```

### 3.2 seqv3 注入（B0/B1/B2 训练数据；`--answer_frames` 恢复 TG 序列归属）
```bash
python scripts/inject_metatoken.py \
    --input ./b4dl_dataset/stage2_full_train_148k.json \
    --ego_meta ./b4dl_dataset/ego_metadata.json \
    --frame_motion ./b4dl_dataset/ego_frame_motion.json \
    --sequence_metadata ../encoders/lidarclip/annotations/sequence_metadata.json \
    --answer_frames \
    --output ./b4dl_dataset/stage2_full_train_seqv3_148k.json
```
两阶段数据同理，把 `--input` 换成 `stage2_train.json` / `stage3_train.json`（输出 `*_seqv3.json`）。
seqv2 同命令去掉 `--answer_frames`；seq v1 去掉 `--frame_motion --sequence_metadata --answer_frames`。

### 3.3 meta2（= 对 3.2 产物重渲染 meta 段，B3 训练数据）
```bash
python scripts/re_render_meta.py \
    --data ./b4dl_dataset/stage2_full_train_seqv3_148k.json \
    --frame_motion ./b4dl_dataset/ego_frame_motion.json \
    --out ./b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json
```

### 3.4 B4a 过采样（从 meta2 再生，秒级）
```bash
python scripts/oversample_tg_highframe.py \
    --data ./b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json \
    --out ./b4dl_dataset/stage2_full_train_seqv3_meta2_oversampled_150k.json
```

> B3 的完整训练配方（含 `--whole_scene` 等训练侧标志）见 `B4DL_B3最优训练方案存档_20260909.md`；
> `--whole_scene` 是**训练/评测侧**开关（`dataset.py` 行为），不属于数据注入。

## 4. 注意事项

1. **历史文档里的命令不再即开即用**：`B4DL_seqv2评测对比与坍缩分析_20260825.md`、`B4DL_两阶段训练评测与混合法决策_20260826.md`、wiki Reproduction-Log 等出现的 `stage2_full_train_seqv2_148k.json` / `stage2_full_train_seqv3_148k.json` 等路径，现在需先按 §3 再生（历史记录本身是「当时做了什么」的存档，未改写）。
2. `scripts/verify.sh`（旧校验脚本）与 `scripts/run_stage2_full_seqv2.sh` 等历史脚本引用的数据已删除，跑之前先再生。
3. 迁移手册 `B4DL_迁移清单与SSH搬运手册_20260910.md` 的 A9 行已同步为清理后体积；其 `tar --exclude='mllm/b4dl_dataset/*.bak*'` 保持幂等（现无 `.bak` 可排除）。
4. 若要再瘦身：跑完 `_vram_smoke_bs8` 后可删 B4a 过采样文件（132M，§3.4 秒级再生）；`test_qa_simple_only.json`（3.9M）亦可按 `test_qa.json` 的 task 过滤重建。
