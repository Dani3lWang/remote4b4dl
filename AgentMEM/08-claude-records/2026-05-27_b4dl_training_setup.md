# B4DL 训练环境搭建与调试记录

**日期**: 2026-05-27  
**任务**: 完成 LiDARCLIP 特征提取后的 B4DL 训练流程搭建、路径调试与 GPU 资源分配

---

## 任务背景

B4DL 是一个基于 4D LiDAR 点云的 LLM 时空理解模型，训练分为三个阶段：

1. **数据生成** → 2. **LiDARCLIP 特征提取** → 3. **VTimeLLM 多模态训练 (Stage1 + Stage2)**

特征提取已完成，位于 `encoders/lidarclip/b4dl/`。本次任务聚焦于 Stage1/Stage2 MLLM 训练的环境配置与调试。

---

## 关键发现与修复

### 1. WandB 配置问题

- **问题**: 训练脚本默认启用 wandb 日志，需要 API 登录
- **处理过程**:
  - 本地 `mllm/wandb/` 目录（之前失败运行残留）与真实 wandb 包冲突，导致 `wandb has no attribute 'api'` 错误
  - 删除本地目录，安装 wandb 包，配置 API Key
  - 用户最终决定取消 wandb：将所有训练脚本的 `--report_to wandb` 改为 `--report_to none`
- **涉及文件**: `scripts/stage1.sh`, `scripts/stage2.sh`, `scripts/stage1_glm.sh`, `scripts/stage2_glm.sh`

### 2. 特征路径命名不匹配（关键 Bug）

- **问题**: Stage1 训练启动后崩溃，报 `FileNotFoundError` 找不到 `{scene_id}.npy`
- **根因**: `mllm/vtimellm/train/dataset.py` 第 404 行按 `scene_id` 查找特征文件，但 `stage1_features/` 中文件按 `frame_id` 命名（逐帧特征），而 `stage2_features/` 才是按 `scene_id` 命名（场景级拼接特征）
- **修复**: Stage1 和 Stage2 训练都需要场景级特征，将两个脚本的 `--feat_folder` 都指向 `stage2_features/`（850 个 .npy 文件，shape (N, 768)）
- **验证**: 所有 699 个 stage1 scene_id 在 stage2_features 中均有匹配文件

### 3. GPU 资源冲突与 OOM

- **问题**: GPU 0 上存在其他 screen 会话的僵尸进程:
  - PID 805702（EIA 训练，screen `792477.lxy`，占用约 16GB）
  - PID 881040（UniDSeg，screen `881032.R073_...`，占用约 4.7GB）
  - 剩余仅 ~11GB，不足以运行 Stage2 训练
- **修复**: 采用方案 A — 将 Stage2 切换到空闲的 GPU 1，修改 `stage2.sh` 中 `gpu_vis=0` → `gpu_vis=1`

### 4. RTX 5090 兼容性

- 没有适用于 sm_120 的 flash_attn，自动回退到 SDPA
- CUDA 13.0, PyTorch 2.11.0, Transformers 5.8.0, DeepSpeed 0.16.4

---

## 文件修改清单

| 文件 | 变更内容 |
|------|----------|
| `mllm/scripts/stage1.sh` | `data_path` 路径修正；`feat_folder` 改为 stage2_features；`report_to` 改为 none |
| `mllm/scripts/stage2.sh` | `gpu_vis` 改为 1；`data_path` 路径修正；`feat_folder` 改为 stage2_features；`report_to` 改为 none |
| `mllm/scripts/stage1_glm.sh` | `report_to` 改为 none |
| `mllm/scripts/stage2_glm.sh` | `report_to` 改为 none |
| `CLAUDE.md` | 新建，规定 Git 提交规范 |

---

## 训练结果

### Stage 1（已完成）

- **状态**: 训练成功
- **Checkpoint**: `mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin`
- **配置**: GPU 1, vicuna-v1-5-7b, 699 条场景级对话, 1 epoch, lr=1e-3

### Stage 2（待执行）

- **启动命令**: `cd /root/autodl-tmp/wql/mmb4dl/mllm && bash scripts/stage2.sh`
- **配置**: GPU 1, LoRA (r=64, alpha=128), 68,695 条 QA 对话, 2 epochs, lr=1e-4
- **注意**: 所有数据均为 "train" 分割，无验证集

---

## 遗留事项

- [ ] Stage2 训练待手动启动
- [ ] `stage1_features/` （29,862 个逐帧特征文件）目前未使用，需确认是否有后续用途
- [ ] Stage2 缺少验证集划分

---

## 环境信息速查

| 项目 | 路径 |
|------|------|
| LiDARCLIP 特征 | `encoders/lidarclip/b4dl/stage2_features/` |
| Stage1 数据 | `mllm/data/stage1_lidarllm_mm.json` |
| Stage2 数据 | `mllm/b4dl_dataset/stage2_conversations.json` |
| Stage1 权重 | `mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/` |
| 训练脚本 | `mllm/scripts/stage*.sh` |
