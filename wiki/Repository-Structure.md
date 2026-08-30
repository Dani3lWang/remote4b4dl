# 目录结构

```
mmb4dl/
├── README.md                    # 官方 README（管线概览、Demo、引用）
├── CLAUDE.md                    # Claude Code 工作指引（项目速查权威文件）
├── requirements.txt             # ⚠️ 已废弃，勿用（版本过新，与 mllm 冲突）
├── smoke_test.log               # 冒烟测试日志
│
├── datageneration/              # 数据生成管线（GPT-4o）
│   ├── config.py                # 全局配置：模型名、环境变量、任务类型、帧采样
│   ├── generate_description.py  # Step 1：相机图 → 场景描述 JSON
│   ├── generate_dataset.py      # Step 2：描述 → 6 类任务 QA JSON
│   ├── prompts.py               # 全部 prompt 模板（1 描述 + 6 QA）
│   ├── utils.py                 # ReadJson / base64 / QA 解析 / conversation 组装
│   ├── scripts/                 # generate_description.sh / generate_dataset.sh
│   └── tools/                   # metadata 构建、stage1 数据转换（LiDAR-LLM→162K）
│
├── encoders/lidarclip/          # LiDAR-CLIP 编码器
│   ├── train.py                 # Lightning 训练（MSE 对齐冻结 CLIP）
│   ├── extract_pc_features.py   # 特征提取（旧，frame_id 键控）
│   ├── extract_pc_features_sample_token.py  # 特征提取（新，sample_token 键控）
│   ├── val_mse_probe.py         # val 场景 MSE 探针（收敛/选型判据）
│   ├── run_anneal_chain.sh      # 退火链编排（tmux b4dlanneal）
│   ├── early_stop_monitor*.py   # 训练早停监控（v2 用 train_loss.csv 真值）
│   ├── validate_encoder_fit.py / validate_scatter.py / smoke_train.py
│   ├── lidarclip/               # 模型包（SST + AttentionPool2d + loader + 兼容层）
│   ├── sst/                     # SST 官方代码（外部依赖，mmdet3d 0.x fork）
│   └── mmdetection3d/           # mmdet3d 源码拷贝（外部依赖）
│
├── mllm/                        # VTimeLLM 训练/推理/评测
│   ├── run_stages.sh            # 一键 Stage1+2
│   ├── run_baseline_eval.sh     # baseline 评测一键脚本
│   ├── requirements.txt         # ★ 权威依赖文件
│   ├── vtimellm/                # 模型包（model/train/inference/demo）
│   ├── scripts/                 # 全部训练/数据/评测驱动脚本 + zero*.json
│   ├── evaluation/              # test_b4dl.py、evaluate_model.py、分析脚本
│   ├── data/                    # 推理 demo 用示例
│   └── training_logs/           # 训练日志（tee）
│
├── docs/
│   ├── B4DL_复现方案.md          # 论文+官方仓库逐行解析的完整复现方案（71KB）
│   ├── B4DL_训练评测报告_20260810.md
│   ├── mmb4dl.pdf               # 论文原文
│   ├── learn docs/              # 16 篇开发记录（审计、基线、分析，见 Reproduction-Log）
│   └── 参考/                    # LiDAR-LLM 参考源码、训练手记
│
├── assets/                      # README 用的图示与 GIF
├── Claude_record/               # 会话记录
└── training_logs/               # 早期训练日志
```

## mllm/scripts/ 速查

| 脚本 | 用途 |
|------|------|
| `stage1.sh / stage2.sh / stage3.sh` | 标准三阶段（deepspeed zero3） |
| `stage1_glm.sh / stage2_glm.sh` | ChatGLM backbone 版 |
| `run_stage2_full_seqv3.sh` | 两阶段法驱动（Phase A → merge → Phase B，幂等） |
| `run_stage2_full_seqv3_mixed.sh` | 混合法驱动（当前基线 B0 方案） |
| `run_stage2_full_seqv3_mixed_b1.sh` | B1 变体（独立 output_dir 保 B0 可比） |
| `run_b1_pipeline.sh` | B1 全流水线（重提特征 → stage1 162K → mixed-b1 → 评测） |
| `merge_stage2.py` | LoRA merge 进 base 保存全量模型 |
| `build_stage2_full_train.py` | HF 官方数据 → 训练格式（148,271 条 + TG 标签） |
| `inject_metatoken.py` | metatoken + feat_indices/feat_range 注入 |
| `generate_ego_metadata.py` / `ego_text.py` | ego 运动元数据生成 / 文本渲染单一来源 |
| `run_b4dl_eval.sh` | 一键评测（划分 + 评测 + 指标） |
| `zero2.json / zero3.json / zero3_offload.json` | DeepSpeed 配置 |

## 大文件存放约定

预训练模型放 `mllm/base_model/`，checkpoint 放 `mllm/checkpoints/`，特征 `.npy` 放各 `b4dl/` 目录，评测数据放 `mllm/b4dl_dataset/` —— 均不入库（.gitignore），关键文件以 MD5 记录在 [[Reproduction-Log]]。
