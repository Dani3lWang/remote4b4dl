# MLLM 后训练验证与 Stage3 准备工作记录

> 日期：2026-05-27 ~ 2026-05-28  
> 分支：remote  
> 背景：Stage1（projector 对齐）和 Stage2（LoRA QA 微调，68,695 条）已完成

---

## 一、Bug 修复

### 1.1 transformers DynamicCache 兼容性

**文件**：`mllm/vtimellm/model/vtimellm_arch.py`

新版 transformers 将 `past_key_values` 从元组改为 `DynamicCache` 对象，不再支持 `[-1][-1]` 索引，导致推理时 `generate()` 崩溃。

**修复**：在访问 `past_key_values` 前先检查是否有 `get_seq_length` 方法（DynamicCache 特征），有则直接调用，否则走旧版元组索引逻辑。

### 1.2 Stage3 训练 DeepSpeed merge_and_unload 失败

**现象**：`deepspeed` 启动 Stage3 训练时，`model.merge_and_unload()` 报错 `The size of tensor a (0) must match the size of tensor b (4096)`。

**排查**：独立 Python 环境中 merge 完全正常，仅在 DeepSpeed launcher 环境下失败。

**解决方案**：新建 `mllm/scripts/merge_stage2.py`，在非 DeepSpeed 环境中预先把 Stage2 LoRA 合并为完整 checkpoint（`checkpoints/vtimellm-vicuna-v1-5-7b-stage2-merged/`），Stage3 训练直接以合并后模型为基座，走标准 LoRA 训练流程。

---

## 二、新建文件清单

| 文件 | 说明 |
|------|------|
| `mllm/scripts/verify_stage2.py` | 快速推理验证，采样 20 条输出 Q/A 对比 |
| `mllm/scripts/create_splits.py` | 按 scene 级别 80/10/10 划分数据 |
| `mllm/scripts/merge_stage2.py` | 预合并 Stage2 LoRA 为完整模型 |
| `mllm/scripts/stage3.sh` | Stage3 训练脚本（使用合并后基座模型） |
| `mllm/scripts/run_b4dl_eval.sh` | 一键编排：划分 → 评测 → 指标 |
| `mllm/vtimellm/eval/b4dl_eval.py` | B4DL 评测推理，输出 JSONL |
| `mllm/vtimellm/eval/b4dl_metrics.py` | 多类型指标计算 |

---

## 三、数据划分

按 scene_id 级别 80/10/10 划分（699 个场景共用于 Stage2 和 Stage3）：

| 集合 | Scene 数 | Stage2 条数 | Stage3 条数 |
|------|----------|-------------|-------------|
| Train | 559 | 54,901 | 63,821 |
| Val | 69 | 6,723 | 7,710 |
| Test | 71 | 7,071 | 8,045 |

划分配置保存在 `b4dl_dataset/split_scenes.json`。

---

## 四、Stage2 测试集评测结果

**7,071 条测试数据，71 个未见过场景**

### 整体

| 指标 | 值 |
|------|-----|
| Overall Exact Match | **64.63%** |

### Binary QA（5,179 条，73.2%）

| 指标 | 值 |
|------|-----|
| Accuracy | 82.80% |
| Precision | 84.02% |
| Recall | 89.72% |
| F1 | 86.78% |

> 模型对 Yes/No 类型的存在性判断表现良好。

### Frame Range（1,405 条，19.9%）

| 指标 | 值 |
|------|-----|
| Exact Match | 8.54% |
| Mean IoU | 17.11% |
| R1@0.5 | 13.95% |
| R1@0.7 | 10.11% |

> 时间定位是明显短板，模型倾向输出 "from frame 000 to frame 008"。

### Categorical（487 条，6.9%）

| 指标 | 值 |
|------|-----|
| Accuracy | 39.63% |
| Macro F1 | 7.07% |

> 物体类别识别较弱，部分罕见类别（如 Wheelchair、BUS vs TRUCK）混淆。

---

## 五、Stage3 训练配置

**基座模型**：`checkpoints/vtimellm-vicuna-v1-5-7b-stage2-merged/`（Stage2 LoRA 已合并）

| 参数 | 值 |
|------|-----|
| 训练数据 | `b4dl_dataset/stage3_train.json`（63,821 条描述性 captioning） |
| Epochs | 3 |
| Learning Rate | 2e-5 |
| LoRA r/alpha | 64/128 |
| Effective Batch Size | 128 (8×16) |
| 特征目录 | `../encoders/lidarclip/b4dl/stage2_features` |
| GPU | 1（RTX 5090 32GB） |
| 输出目录 | `checkpoints/vtimellm-vicuna-v1-5-7b-stage3/` |

---

## 六、继续工作的命令

```bash
# Stage3 训练
cd /root/autodl-tmp/wql/mmb4dl/mllm
bash scripts/stage3.sh

# Stage3 训练完成后评测
bash scripts/run_b4dl_eval.sh --stage3
```

---

## 七、Git 提交记录

```
a1cae9f 修复 Stage3 训练 DeepSpeed ZeRO-3 环境下 LoRA merge 失败问题
3705d5a 添加 MLLM 后训练验证与评估体系
```
