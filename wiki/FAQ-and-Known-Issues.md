# 已知问题与注意事项

按模块整理的踩坑清单。多数问题已在代码中修复，此处保留背景与检查方法。

## 环境

| 问题 | 说明 |
|------|------|
| 依赖文件选择 | 只用 `mllm/requirements.txt`；根目录 `requirements.txt` 版本过新且冲突 |
| peft 0.4.0 | Stage3 的 merge + 重加 LoRA 模式依赖旧式 API，不可升级 |
| transformers 4.31.0 | monkey-patch 与旧版 generation API 绑定；`merge_stage2.py` 里已专门修了新版 generation_config 校验报错的兼容分支 |
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
- **旧特征全部作废**：现存 stage1/stage2 特征均为 ONCE 旧编码器产物，编码器定稿后必须全量重提，不可增量混提
- 单卡训练需去掉 ddp（与 fork dataloader 互锁会卡死）
- loss 真值看 `logs/train_loss.csv`，不要信 wandb offline
- ViT-L/14 + batch 32 是 32GB 显存上限，超了用梯度累积或 ViT-B/32

## 训练（mllm/）

- **特征文件名 = 数据 JSON 的 `scene_id`**（`{feat_folder}/{scene_id}.npy`），两套数据必须同代际：新 stage1 数据（sample_token 键控）配 `stage1_features_sample/`，旧数据配 `stage1_features/`
- `LazySupervisedDataset` 旧版在特征缺失时会随机替换样本（静默数据丢失）——现版已改启动时 fail-fast，但检查日志异常打印仍是好习惯
- DeepSpeed no_sync 崩溃由 train.py 头部 monkey-patch 修复，勿删除
- Stage3 的 `--model_name_or_path` 必须指向 **merged 全量模型**，不是 stage2 LoRA 目录
- 训练/评测数据必须同代际（seqv3 训练的模型评测必须带全参数），见 [[Inference-and-Evaluation]] 的代际表
- time_grounding 类问题问题文本无帧号，需 `--answer_frames` 恢复归属（oracle，须声明）

## 推理与评测

- `demo_gradio.py` 的 `gr.Examples` 引用未定义的 `root_dir` 会 NameError，需手动修正；所有路径显式传绝对值
- 评测大文件（test_qa.json 等）不在仓库内，在远端训练机；替换任何组件前先核对 [[Reproduction-Log]] 的 MD5
- 指标对比必须用冻结口径（pycocoevalcap 语料级 BLEU / Meteor-1.5 jar / roberta-large L17 BERTScore），NLTK 后端数值不可比

## 论文 vs 复现的已知差异

- 编码器权重：官方从未发布（"You need to train the model first"），本地旧 ckpt 是 ONCE 权重（domain gap），自训 nuScenes 编码器（B2）进行中
- METEOR 0.1729 vs 论文 0.275：修正口径后仍偏低，是唯一遗留的口径考证项
- mIoU 0.2696 vs 论文 0.311：TG 时间定位是当前最大差距项（曾出现 (0,8) 占 87% 的坍缩，seqv3 + answer_frames 后缓解）
- 论文报告的 benchmark 未提供每条 QA 的序列归属字段，TG 的包含序列靠 GT 答案帧范围回退解析（双侧一致，2783/2783）
