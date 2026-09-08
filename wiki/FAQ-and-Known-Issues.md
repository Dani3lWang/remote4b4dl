# 已知问题与注意事项

按模块整理的踩坑清单。多数问题已在代码中修复，此处保留背景与检查方法。

## 环境

| 问题 | 说明 |
|------|------|
| 依赖文件选择 | 只用 `mllm/requirements.txt`；根目录 `requirements.txt` 版本过新且冲突 |
| 版本基线（2026-09-08 实测） | `wqlc` 环境实为 **transformers 4.47.0 / torch 2.8.0+cu128 / peft 0.13.2** / deepspeed 0.16.4（CLAUDE.md 旧清单已同步修正）；早期文档的 4.31.0/0.4.0 是历史版本，DoRA/NEFTune/现代 logprob API 在现版本均可用 |
| peft API | 0.13 下 `merge_and_unload`/`get_peft_model`/`LoraConfig(use_dora=True)` 已实测可用；若旧代码报 merge API 错先核对环境版本 |
| flash-attn 与 RTX 5090 | sm_120 不支持 flash-attn 2.x，`train_mem.py` ImportError 时自动回退 SDPA |
| METEOR 需要 java | `apt-get install default-jre-headless`；缺 jar 会静默回退 NLTK（与论文不可比），看结果 JSON 的 `metric_backend` 字段确认 |
| clip.load 不转 fp16 | train.py 已加 `clip.model.convert_weights()` |

## 数据生成（datageneration/）

- `--start_index/--end_index` 必须是 `SAVE_TERM`(10) 的倍数，否则直接退出
- `--api_key` 必填（或设 `OPENAI_API_KEY`）；历史提交中有旧 key，公开发布前需清理 git 历史
- time_grounding 答案有格式过滤损失（不含 `from frame` 即丢弃）
- 输出文件名由 `START_INDEX + i*SAVE_TERM` 推算且按文件名末尾数字排序读取，**不可改名**
- `tools/generate_stage1_caption.py` 与 `tools/create_metadata.py` 引用了 config.py 中不存在的配置项，直接运行会 AttributeError（仅作参考文档用）

## LiDAR-CLIP 编码器

- checkpoint 加载必须 `torch.load(..., weights_only=False)`（PL ckpt 含调度器等非张量对象）；`strict=False` 只为忽略旧版 bbox_head 残留键，**missing_keys 非空必须报错**
- 提特征必须 `eval()`；探针反而必须 train() 模式（与提特征约定一致才可比）
- **旧特征已全量替换**（2026-09-06 完成）：当前 stage1/stage2 特征全部出自退火定稿的 nuScenes 自训编码器（28,130 帧 + 850 场景），旧 ONCE 特征备份在 `b4dl/*_once` 目录；新旧编码器特征**不可增量混提**，见 [[LiDAR-CLIP-Encoder]]
- 单卡训练需去掉 ddp（与 fork dataloader 互锁会卡死）
- loss 真值看 `logs/train_loss.csv`，不要信 wandb offline
- ViT-L/14 + batch 32 是 32GB 显存上限，超了用梯度累积或 ViT-B/32

## 训练（mllm/）

- **特征文件名 = 数据 JSON 的 `scene_id`**（`{feat_folder}/{scene_id}.npy`），两套数据必须同代际：新 stage1 数据（sample_token 键控）配 `stage1_features_sample/`，旧数据配 `stage1_features/`
- `LazySupervisedDataset` 旧版在特征缺失时会随机替换样本（静默数据丢失）——现版已改启动时 fail-fast，但检查日志异常打印仍是好习惯
- DeepSpeed no_sync 崩溃由 train.py 头部 monkey-patch 修复，勿删除
- Stage3 的 `--model_name_or_path` 必须指向 **merged 全量模型**，不是 stage2 LoRA 目录
- 训练/评测数据必须同代际：seqv3 代际（B0/B1）评测带 `--per_sequence --frame_motion --sequence_metadata --answer_frames` 全参数；B2+ 整场景代际（B2/B3/B4a）带 `--whole_scene --per_sequence --answer_frames`，见 [[Inference-and-Evaluation]] 的代际表
- time_grounding 类问题问题文本无帧号，需 `--answer_frames` 恢复归属（oracle，须声明）

## 推理与评测

- `demo_gradio.py` 的 `gr.Examples` 引用未定义的 `root_dir` 会 NameError，需手动修正；所有路径显式传绝对值
- 评测大文件（test_qa.json 等）不在仓库内，在远端训练机；替换任何组件前先核对 [[Reproduction-Log]] 的 MD5
- 指标对比必须用冻结口径：pycocoevalcap 语料级 BLEU-4 / **METEOR 主口径 NLTK-2005（2026-09-07 起，jar 仅作衔接旧表的参考）** / roberta-large 第 17 层 BERTScore；与论文对比一律用 `meteor`（NLTK），禁止混用旧口径数值
- 全量评测（30,145 条）单次可能触发超时：B4a 管线阶段 2 首尝试 2026-09-08 以 rc=124 超时退出，同日手动补跑 18:18 完成；评测自带 ckpt 断点续跑，超时后同命令重跑即续

## 论文 vs 复现的已知差异（2026-09-08 现状）

- 编码器权重：官方从未发布（"You need to train the model first"），本地旧 ckpt 是 ONCE 权重（domain gap）——✅ 已解决：nuScenes 自训 + 退火 3ep 定稿（val MSE 0.0992），B1 起特征全量重提
- METEOR：✅ 已闭环（2026-09-07 溯源）：论文引用 [2] = Banerjee & Lavie 2005，NLTK-2005 是其忠实实现；该口径 B0-B3/B4a = 0.3344~0.3378 全超论文 0.275（论文 0.275 的精确实现不可考，变体区间 0.14~0.38）
- mIoU：✅ 主指标已超论文（B3 0.3467 vs 0.311）；当前最大差距转 TG 高帧段——GT start≥25 的 413 条测试样本 B3 预测落回该段的仅 13 条（≈3%，GT 占 14.8%），且整体 −3.9 帧偏早；B4a 高帧段过采样实验（2026-09-08 完结）为**负结果**（mIoU Δ-0.0196 显著回退），病灶指向"输入无显式帧号信号"，下一步做帧身份锚点消融（见 [[Reproduction-Log]] 路线图）
- 论文报告的 benchmark 未提供每条 QA 的序列归属字段，TG 的包含序列靠 GT 答案帧范围回退解析（双侧一致，2783/2783）
