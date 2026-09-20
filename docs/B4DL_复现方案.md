# B4DL 论文复现方案

> 论文：**B4DL: A Benchmark for 4D LiDAR LLM in Spatio-Temporal Understanding**（ACM MM 2025，arXiv:2508.05269）
> 原始仓库：`D:\tmp\B4DL`（GitHub: ccho4702/B4DL）
> 修改版仓库：`D:\tmp\remote4b4dl`（GitHub: Dani3lWang/remote4b4dl）——修复了多项 bug，添加了内置评测代码、Stage 3 训练脚本、Gradio Demo，并采用单环境方案
> 本方案基于对两个仓库全部代码与论文的逐行解析编写，包含完整的步骤、命令、超参数，以及**仓库代码与论文不一致的关键坑位清单**。下方标注【remote4b4dl】的内容为修改版仓库独有或已修复的部分。

---

## 一、论文与仓库总览

### 1.1 论文做了什么

B4DL 提出了针对 **4D LiDAR（时序点云序列）** 的多模态大语言模型（MLLM）基准，包含三部分贡献：

1. **B4DL 基准与数据集**：6 个任务（Existence、Binary QA、Time Grounding、Description、Temporal Understanding、Comprehensive Reasoning），基于 nuScenes 的 850 个 scene（700 train / 150 test）生成 178.4k 条 QA 对。
2. **数据生成管线**：两步式——①用 GPT-4o 从与 LiDAR 时间对齐的 6 路环视图像提取 4D LiDAR 上下文描述；②将描述 + 人工标注（HA）转换为指令式 QA。
3. **B4DL 模型**：基于 VTimeLLM 架构改造的 MLLM，包含 LiDAR 编码器（LiDARCLIP/SST）、LiDAR Aligner（线性投影）、Metatoken 三个对齐模块，采用两阶段训练（3D 理解 → 4D 理解）。

### 1.2 复现目标（论文中的可量化结果）

| 目标 | 来源 | 内容 |
|---|---|---|
| 主结果 | Table 3 | B4DL vs B4DL-LiDARLLM vs VTimeLLM 在 7 个指标上的对比 |
| 消融实验 | Table 4 | Human Annotation (HA) × Metatoken 的 4 组消融 |
| 数据规模 | Table 6 | 10% / 25% / 50% / 75% / 100% 训练数据量 |
| 数据多样性 | Table 5 | DCScore 对比（可选） |
| 跨数据集 | Table 3 † | 在 Waymo 1k 子集上零样本推理（可选，仓库无现成代码） |

**Table 3 目标数值（B4DL 模型，nuScenes 测试集 900 个序列）**：

| Accuracy↑ | mIoU↑ | B@4↑ | ROUGE-L↑ | METEOR↑ | BERTScore↑ | GPT Score↑ |
|---|---|---|---|---|---|---|
| 0.762 | 0.311 | 0.095 | 0.322 | 0.275 | 0.897 | 59.513 |

### 1.3 仓库结构与论文对应关系

| 目录 | 功能 | 对应论文章节 |
|---|---|---|
| `datageneration/` | GPT-4o 数据生成管线（描述提取 + QA 转换） | §3.2, Appendix F |
| `datageneration/data/metadata/` | 仓库自带 850 scene / 5100 sequence 的划分元数据（74MB） | §3.3 |
| `encoders/lidarclip/` | LiDAR 编码器 E_L：SST 骨干 + AttentionPool，CLIP 对齐训练 | §4.1 |
| `encoders/lidarclip/sst/` | TuSimple SST 仓库 fork（内含魔改 mmdet3d 0.15.0） | §4.1 |
| `encoders/lidarclip/extract_description.py` | 基于 nuScenes 标注的规则化运动描述 GT（即 HA 来源） | §3.2 |
| `mllm/` | B4DL 模型训练/推理框架（VTimeLLM fork 改造） | §4 |
| `mllm/scripts/stage1.sh / stage2.sh` | 两阶段训练脚本 | §4.2 |
| `mmb4dl/` | 论文 PDF 与 MD | — |

### 1.4 整体复现流水线

```
nuScenes (v1.0-trainval)
    │
    ├─[datageneration] GPT-4o ──→ B4DL 数据集 (178k QA)          ← 或直接下载 HF 数据集
    │
    ├─[encoders/lidarclip] train.py ──→ LiDAR-CLIP 编码器权重
    │        └─ extract_pc_features.py ──→ stage1/stage2 点云特征 (.npy)
    │
    └─[mllm]
         ├─ Stage 1: LiDAR-LLM 数据 + stage1 特征 → 训练 mm_projector
         └─ Stage 2: B4DL 数据 + stage2 特征 → 训练 LoRA
                    │
                    └─ 推理 + 六任务评测 → Table 3 / Table 4
```

---

## 二、资源清单（开始之前必须备齐）

### 2.1 硬件

| 项目 | 要求 | 说明 |
|---|---|---|
| GPU | 1× RTX 4090 (24GB) 或更高 | 论文明确：单卡 4090，全部实验 24h 内完成 |
| 显存 | ≥24GB | Stage2 用 ZeRO-3 + LoRA + 梯度检查点，7B 模型单卡可跑 |
| 磁盘 | ≥500GB | nuScenes full ~350GB + 权重/特征/checkpoint ~100GB |
| 系统 | **Linux**（Ubuntu 20.04/22.04） | deepspeed、flash-attn、mmcv-full 编译均不原生支持 Windows。当前机器是 Windows，建议使用 WSL2 + Docker，或租用云 GPU（AutoDL/阿里云等） |

### 2.2 数据集

| 数据 | 获取方式 | 用途 |
|---|---|---|
| nuScenes v1.0-trainval (full) | https://www.nuscenes.org 注册下载（~350GB） | 全流程基础数据 |
| nuScenes v1.0-mini | 同上（~4GB） | 冒烟测试用 |
| B4DL 数据集（178k QA） | **方案 A**：HuggingFace `ccho4702/nuScenes-B4DL` 直接下载；**方案 B**：用 `datageneration/` 自己生成 | Stage2 训练 + 测试集 |
| LiDAR-LLM Nu-Caption（162k QA） | LiDAR-LLM 官方仓库（github.com/senqiaoyang/LiDAR-LLM）提供的下载链接 | Stage1 训练 |
| Waymo Open Dataset（可选） | https://waymo.com/open | 跨数据集泛化实验（§5.3） |

#### HuggingFace 数据集详细结构（`ccho4702/nuScenes-B4DL`）

下载命令：

```bash
huggingface-cli download ccho4702/nuScenes-B4DL --repo-type dataset --local-dir ./b4dl_hf
```

下载后的目录结构：

```
b4dl_hf/
├── dataset/
│   ├── train/
│   │   ├── stage2.json          # 68,695 条（Stage 2 训练数据，raw 格式）
│   │   └── stage3.json          # 79,576 条（Stage 3 训练数据，raw 格式）
│   └── test/
│       ├── existence.json              # 3,770 条
│       ├── binary.json                  # 7,525 条
│       ├── time_grounding.json         # 2,783 条
│       ├── description.json            # 3,770 条
│       ├── temporal_understanding.json # 4,757 条
│       └── comprehensive.json           # 7,540 条
└── metadata/
    ├── scene_metadata.json      # 850 场景元数据（700 train + 150 val/test）
    └── sequence_metadata.json   # 5100 序列元数据（4200 train + 900 val/test）
```

**数据量验证**（与论文 Table 2 逐项比对）：

| 项目 | 论文值 | HF 实际值 | 匹配 |
|---|---|---|---|
| 训练集 stage2 | — | 68,695 | — |
| 训练集 stage3 | — | 79,576 | — |
| 训练集合计 | 148,255 | 148,271（stage2+stage3） | ✓（差 16 条为过滤噪声） |
| 测试集 Existence | 3,770 | 3,770 | ✓ |
| 测试集 Binary QA | 7,525 | 7,525 | ✓ |
| 测试集 Time Grounding | 2,783 | 2,783 | ✓ |
| 测试集 Description | 3,770 | 3,770 | ✓ |
| 测试集 Temporal Understanding | 4,757 | 4,757 | ✓ |
| 测试集 Comprehensive | 7,540 | 7,540 | ✓ |
| 测试集合计 | 30,145 | 30,145 | ✓ |
| 场景数 | 850（700/150） | 850（700/150） | ✓ |
| 序列数 | 5100（4200/900） | 5100（4200/900） | ✓ |

**数据格式说明**：HF 数据集的 JSON 是 **raw 格式**（`question`/`answer` 字段），不是训练脚本直接需要的 VTimeLLM conversations 格式。每条数据结构如下：

```json
// train/stage2.json 的单条数据（raw 格式）
{
  "split": "train",
  "scene_token": "2ffd7e2a1daf4b928464ddb2ed3dca59",
  "scene_id": "003833660",
  "human_annotation": "Densely parked trucks",
  "question": "Was a pedestrian present in front of the ego vehicle between frame 30 and frame 38?",
  "answer": "Yes."
}
```

训练脚本（`stage2.sh`）需要的是 **conversations 格式**：

```json
// stage2_conversations.json 的单条数据（训练用格式）
{
  "scene_id": "003833660",
  "scene_token": "2ffd7e2a1daf4b928464ddb2ed3dca59",
  "split": "train",
  "conversations": [
    {"from": "human", "value": "<video>\nWas a pedestrian present in front of the ego vehicle between frame 30 and frame 38?"},
    {"from": "gpt",   "value": "Yes."}
  ]
}
```

**格式转换**：需要把 raw 格式转换为 conversations 格式。转换逻辑：把 `question` 包进 `{"from":"human","value":"<video>\n"+question}`，把 `answer` 包进 `{"from":"gpt","value":answer}`，组装成 `conversations` 数组。【remote4b4dl】仓库自带的 `mllm/b4dl_dataset/stage2_conversations.json` 已经是转换后的格式（68,695 条），可直接使用。测试集同样需要转换，或使用仓库自带的 `mllm/evaluation/build_test_split.py` 脚本从生成的数据中自动提取并格式化。

**测试集构建**：HF 数据集的测试集按任务分文件存放（6 个 JSON），评测时需要合并为一个统一的 `test_qa.json`。【remote4b4dl】仓库的 `mllm/evaluation/build_test_split.py` 可以完成此工作：

```bash
cd mllm
python evaluation/build_test_split.py \
    --predictions_dir ../b4dl_hf/dataset/test \
    --scene_metadata ../b4dl_hf/metadata/scene_metadata.json \
    --nuscenes_root /path/to/nuScenes \
    --output ./b4dl_dataset/stage2_test.json
```

**注意**：HF 数据集**不包含** Stage 1 训练数据（LiDAR-LLM Nu-Caption），也**不包含**预提取的 .npy 特征文件和 nuScenes 原始数据。这些需要另外获取。

### 2.3 模型权重

| 权重 | 获取方式 | 放置位置 |
|---|---|---|
| Vicuna-7b-v1.5 | HuggingFace `lmsys/vicuna-7b-v1.5`（~13.5GB） | `mllm/base_model/vicuna-v1-5-7b/` |
| CLIP ViT-L/14 | `clip.load("ViT-L/14")` 自动下载；或手动下载 ViT-L-14.pt | 缓存目录或 `encoders/lidarclip/pretrained/` |

### 2.4 API Key

| Key | 用途 | 成本估计 |
|---|---|---|
| OpenAI API（GPT-4o） | ①数据生成（若不直接下载 HF 数据集）；②GPT Score 评测 | 数据生成：5100 序列 × (2 次描述 + 5 次任务 QA) ≈ 3.6 万次调用，含多图输入，估计 **$150–400**；GPT Score 评测：30k 测试样本 × 每条约 $0.01–0.03 ≈ $300–900（可用 gpt-4o-mini 降本做冒烟） |

---

## 三、环境配置

### 方案选择

原始仓库三个子模块的依赖互不兼容，需要两套 conda 环境（见 §3.1/3.2）。【remote4b4dl】修改版仓库已合并为**单环境方案**（`wqlc`），适合不想折腾环境隔离的用户。

| 方案 | 适用仓库 | 环境数 | 优点 | 缺点 |
|---|---|---|---|---|
| 方案 A：双环境 | 原始 B4DL | 2（b4dl_enc + b4dl_mllm） | 严格隔离，依赖不冲突 | 编码器环境 mmcv-full 1.3.9 极难装 |
| 方案 B：单环境 | remote4b4dl | 1（wqlc） | 简单省事，编码器/MLLM 通用 | mmcv/mmdet3d 可能与新版 torch 有兼容问题 |

### 3.0【remote4b4dl】方案 B：单环境（wqlc）

```bash
conda create -n wqlc python=3.10 -y
conda activate wqlc

# PyTorch 2.5.1 + CUDA 12.4
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 核心依赖
pip install transformers==4.31.0 deepspeed==0.16.4 peft==0.4.0 \
            einops wandb tqdm requests easydict \
            "numpy<2" sentencepiece

# flash-attn（RTX 5090 sm_120 不支持 flash-attn，可跳过，训练自动 fallback 到 SDPA）
# 非 5090 卡可尝试安装：
# pip install flash-attn==2.7.0 --no-build-isolation

# 评测依赖
pip install bert-score==0.3.13 pycocoevalcap==1.2 nltk rouge-score openai

# nuScenes 开发包
pip install nuscenes-devkit pyquaternion

# CLIP
pip install openai-clip

# SST 编码器依赖（需要 CUDA 编译环境）
# 参考 §3.1 的 mmcv-full 安装步骤，或使用 remote4b4dl 的 mmdetection3d 目录
cd encoders/lidarclip
pip install -e sst/   # 编译 CUDA 算子，耗时 10-30 分钟
```

> 注：remote4b4dl 仓库的 `CLAUDE.md` 记录了 RTX 5090 + CUDA 13 环境的特殊处理方式（flash-attn 不安装，PEFT 0.13.2 等），详见仓库内 `requirements_sum/` 目录。

### 3.1 环境 A：LiDAR 编码器（encoders/lidarclip）

```bash
conda create -n b4dl_enc python=3.8 -y
conda activate b4dl_enc

# 1) PyTorch 1.9 + CUDA 11.1（与 mmcv-full 1.3.9 预编译 wheel 匹配）
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# 2) OpenMMLab 全家桶（版本必须严格匹配 sst/mmdet3d/__init__.py 的断言）
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0 mmsegmentation==0.14.1 mmengine==0.10.6

# 3) 安装 fork 的 mmdet3d 0.15（即 sst/ 目录，需要 CUDA 编译，耗时 10–30 分钟）
cd encoders/lidarclip
pip install -e sst/

# 4) 其余依赖
pip install pytorch-lightning==1.6.2 nuscenes-devkit==1.1.10 openai-clip einops \
            pyquaternion tqdm matplotlib==3.5.2 Pillow==9.5.0 "numpy<1.24" wandb open3d
```

**坑位提示**：
- `mmcv-full==1.3.9` 必须用与你 torch/CUDA 版本匹配的预编译 wheel，否则源码编译极易失败。
- `sst/` 安装时会编译 voxel、roiaware_pool3d、spconv 等 CUDA 算子，确保 `nvcc` 可用。
- 仓库 requirements 里 `numba==0.48.0` 与 `numpy 1.20` 冲突，建议 `numba>=0.56` + `numpy<1.24`。
- `tensorflow`、`waymo_open_dataset`、`lyft_dataset_sdk` 是 SST 仓库遗留依赖，nuScenes 流程**不需要**装。

### 3.2 环境 B：MLLM 训练（mllm）

```bash
conda create -n b4dl_mllm python=3.10 -y
conda activate b4dl_mllm

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.37.2 tokenizers sentencepiece \
            deepspeed==0.14.4 peft==0.4.0 accelerate \
            einops wandb tqdm requests easydict \
            "numpy<2"                              # ← 关键！numpy 2.x 与旧 transformers 不兼容
pip install flash-attn==2.6.3 --no-build-isolation # 编译约 20–60 分钟；或找匹配 wheel
# 评测依赖
pip install bert-score==0.3.13 pycocoevalcap==1.2 nltk rouge-score openai
```

**坑位提示**：
- 仓库 `mllm/requirements.txt` 写 `numpy==2.3.5`，但 `transformers==4.31` 用到 `np.float` 等已删除 API，**实际必须用 numpy<2**（这是仓库 requirements 自身的 bug）。
- 仓库同时出现 `transformers==4.31.0`（mllm/requirements）与 `4.37.2`（根目录 requirements），实测 4.31–4.37 区间均可，与 peft 0.4 搭配推荐 4.31。
- `flash-attn` 编译失败时可改走 `vtimellm/train/train.py`（不带 flash-attn monkey patch 的入口），代价是训练略慢、显存略高。
- 训练脚本带 `--report_to wandb`，需要 `wandb login`，或改成 `--report_to none`。

---

## 四、复现步骤

### Step 0：数据与权重就位

```bash
# nuScenes 目录结构应如下（标准官方结构）：
# /path/nuScenes/
#   ├── samples/   (LIDAR_TOP/*.pcd.bin, CAM_FRONT/*.jpg, ...)
#   ├── sweeps/
#   └── v1.0-trainval/  (*.json 元数据)

# Vicuna 放置：
mkdir -p mllm/base_model
# 将下载的 vicuna-7b-v1.5 放入 mllm/base_model/vicuna-v1-5-7b/
```

验证：`python -c "from nuscenes.nuscenes import NuScenes; NuScenes(version='v1.0-trainval', dataroot='/path/nuScenes', verbose=True)"`

### Step 1：构建 B4DL 数据集

#### 方案 A（推荐，省时省钱）：直接下载 HuggingFace 数据集

```bash
pip install huggingface_hub
huggingface-cli download ccho4702/nuScenes-B4DL --repo-type dataset --local-dir ./b4dl_hf
```

下载后的目录结构与数据量验证详见 §2.2「HuggingFace 数据集详细结构」。

**整理为训练格式**：HF 数据集是 raw 格式（`question`/`answer` 字段），训练脚本需要 conversations 格式。两种做法：

1. **直接使用 remote4b4dl 仓库自带的转换后文件**：`mllm/b4dl_dataset/stage2_conversations.json`（68,695 条，已含 `<video>` 前缀）。但注意这只有 68,695 条训练数据（stage2.json），如果要用全量 148k 需要把 stage3.json 也转换。
2. **自己转换**：把 raw 格式的 `question`/`answer` 转为 conversations 格式（首条 human 消息的 value 须以 `<video>\n` 开头）。转换逻辑见 §2.2 中的格式示例。

**构建测试集**：HF 数据集的测试集按 6 个任务分文件存放。使用 `mllm/evaluation/build_test_split.py` 合并为统一测试文件（命令见 §2.2）。

#### 方案 B：自己跑生成管线（完整复现论文 §3.2）

**Step 1.1 — 4D LiDAR Context Extraction（生成场景描述）**

```bash
cd datageneration
# 分块运行：start/end 必须是 SAVE_TERM=10 的倍数，全量 0→5100
python3 generate_description.py \
    --start_index 0 --end_index 5100 \
    --api_key $OPENAI_API_KEY \
    --nuscenes_root /path/nuScenes \
    --dataroot ./data
```

逻辑说明：每个 sequence（5100 个，每个 3–10 帧、2 keyframe 间隔）取 6 路环视图像，按前向（FRONT/FRONT_LEFT/FRONT_RIGHT）与后向（BACK/BACK_LEFT/BACK_RIGHT）分两组送 GPT-4o，各生成一段三段式描述（[1] 场景 [2] 时序变化 [3] 驾驶员视角），输出 `data/generated_description/generated_description_{start}_{end}.json`。

⚠️ **必须修 bug**：`generate_description.py` 第 140 行有一个 `break`，导致每次运行只处理第一个 chunk（10 条序列）。要么删除该 `break`，要么外层写循环按 10 为步长反复调用（`--start_index 0 --end_index 10`、`10→20`……共 510 次）。

**Step 1.2 — 生成人工标注式 GT 描述（HA，可选但推荐）**

```bash
cd encoders/lidarclip
python extract_description.py \
    --data-path /path/nuScenes \
    --save-dir ./GT_annotations
```

该脚本用规则从 nuScenes 标注生成运动描述（如 *"A car overtakes the ego vehicle between Frame 003 and Frame 017."*），对应论文中的 Human Annotation。注意其默认路径硬编码为 `/home/youngwoo.shin/...`，必须用命令行参数覆盖。

**Step 1.3 — Context-to-QA Transformation（生成六任务 QA）**

```bash
cd datageneration
for task in existence binary description temporal comprehensive; do
python3 generate_dataset.py \
    --start_index 0 --end_index 5100 \
    --api_key $OPENAI_API_KEY \
    --dataroot ./data \
    --task $task
done
```

⚠️ 注意三个问题：
1. **没有 time_grounding 生成代码**：`generate_dataset.py` 只支持 existence/binary/description/temporal/comprehensive 五种。Time Grounding 任务的 QA 需要自行构造——可基于 Step 1.2 的规则化运动描述（天然含 "between Frame X and Frame Y"）转成问答，或新增一个 GPT prompt。
2. **HA 分支有 NameError bug**：`prompts.py` 中各 `generate_*_dataset_prompt` 函数在 `gt_caption` 为真时引用了未赋值的变量（如 `DESCRIPTION_DATASET_PROMPT = DESCRIPTION_DATASET_PROMPT + ...`）。若要复现 Table 4 的 HA 消融，需把各分支修正为"基础 prompt + GT 描述注入"。
3. **任务名不一致**：`config.py` 定义 `temporal_understanding`，而 `generate_dataset.py` 判断的是 `temporal`，整理数据时注意对齐。

**Step 1.4 — 整理为 stage2.json**

将六个任务的输出合并，并转换为训练格式（见 §4.6）。

### Step 2：训练 LiDAR-CLIP 编码器 E_L

```bash
cd encoders/lidarclip   # 使用环境 A
python train.py --name=lidarclip \
    --checkpoint-save-dir=./ckpt \
    --batch-size 128 --workers 4 \
    --data-dir /path/nuScenes \
    --clip-model ViT-L/14
```

训练细节（`train.py` 实测）：

| 项 | 值 |
|---|---|
| 损失 | `F.mse_loss(z_I, z_L)`，即论文公式 (1) 的 L_similarity，只用正样本对 |
| CLIP | ViT-L/14，全程冻结（`torch.no_grad` 编码图像） |
| 可训练参数 | 仅 lidar_encoder（SST + AttentionPool） |
| 优化器 | Adam lr=1e-5 + OneCycleLR（max_lr=1e-3, pct_start=0.1） |
| Epochs | 20 |
| 精度 | fp16 |
| 数据划分 | 自动剔除 150 个 val scene（`loader.py` 按 `sequence_metadata.json` 末 900 条过滤），仅用 700 train scene，防止泄漏 |

⚠️ **必须修 bug**：`lidarclip/loader.py` 第 106 行硬编码了 `/home/youngwoo.shin/lidarclip/annotations/sequence_metadata.json`，改成本仓库的 `annotations/sequence_metadata.json` 实际路径。

模型结构：点云（KITTI 风格坐标，强度/255）→ 动态体素化（voxel 0.5×0.5×6m，范围 x∈[0,40], y∈[-20,20], z∈[-2,4]，80×80 BEV）→ DynamicVFE → SSTInputLayerV2 → SSTv2（4 个 shifted-window sparse attention block，d=128）→ `recover_bev` 得 (B,128,80,80) → AttentionPool2d（128→768 投影 + 8 头 cross-attn）→ **每帧输出 1 个 768 维全局 token**。

预计耗时：约 16.8 万 image-LiDAR 对 × 20 epoch ÷ 128 batch ≈ 26k steps，单卡 4090 约 1–3 天。

### Step 3：提取点云特征

提取 .npy 特征分为两步：先用 nuScenes 点云训练 LiDAR-CLIP 编码器得到 checkpoint（Step 2），再用 checkpoint 提取特征。这里描述第二步。

```bash
cd encoders/lidarclip
python extract_pc_features.py \
    --checkpoint ./ckpt/vit_l_14.ckpt \
    --data-path /path/to/nuScenes \
    --scene-json-path ./annotations/scene_metadata.json \
    --frame-json-path ./annotations/sequence_metadata.json \
    --stage1-save-dir ./b4dl/stage1_features/ \
    --stage2-save-dir ./b4dl/stage2_features/
```

输出：

| 目录 | 文件 | 形状 | 用途 |
|---|---|---|---|
| stage1_features | `{frame_id}.npy` | (1, 768) | Stage 1 训练（单帧 3D） |
| stage2_features | `{scene_id}.npy` | (num_frames, 768)，num_frames≈40 | Stage 2 训练（整 scene 的 4D 序列，即论文公式 (3) 的 cls token 拼接） |

最终生成 850 个 `stage2_features/{scene_id}.npy` 文件（每个 scene 一个）。文件名（如 `003833660.npy`）必须与训练 JSON 中的 `scene_id` 字段一致。Stage 1 和 Stage 2 训练脚本都使用 `stage2_features` 目录。

⚠️ 注意：`--data-path` 必须显式传（默认值是占位符）；`--checkpoint` 默认值也是占位符；脚本运行前会**清空重建**两个输出目录（`create_clean_directory` 会先 `shutil.rmtree` 再 `os.makedirs`）。

#### 提取流程详解（`extract_pc_features.py` 工作原理）

1. **加载编码器**：从 .ckpt（PyTorch Lightning 格式）中提取 `lidar_encoder.*` 参数，加载到 `LidarEncoderSST` 模型。由于旧版 checkpoint 可能含 `bbox_head` 等 key，用 `strict=False` 忽略。加载时需 `weights_only=False`（因 checkpoint 含 scheduler 等非 tensor 对象）。
2. **逐帧推理**：数据加载器遍历 nuScenes 点云，每帧通过 SST + AttentionPool2d 编码为 768 维向量，缓存在内存字典 `lidar_dict[full_path]` 中。
3. **Stage 1 保存**：遍历 `sequence_metadata.json`，按帧保存，每帧一个 `(1, 768)` 的 .npy 文件。
4. **Stage 2 保存**：遍历 `scene_metadata.json`，把每个 scene 的所有帧特征 `torch.cat` 拼接，保存为一个 `(num_frames, 768)` 的 .npy 文件。

#### ⚠️【严重】必须修复 loader.py 的测试场景过滤

`lidarclip/loader.py` 第 104–120 行有一段过滤代码，把最后 900 条序列（150 个 val/test scene）从数据加载器中排除，目的是训练编码器时防止数据泄漏：

```python
# lidarclip/loader.py 第 104-120 行（原始代码）
# ################# for 700 scenes only #################
seq_meta_path = osp.join(osp.dirname(__file__), "..", "annotations", "sequence_metadata.json")
with open(seq_meta_path, "r") as f:
    seq_data = json.load(f)
filtered_scene_list = []
for seq in seq_data[-900:]:              # ← 取最后 900 条（测试场景）
    filtered_scene_list.append(seq["scene_token"])
filtered_scene_list = list(set(filtered_scene_list))
filtered_ok_scene_tokens = [x for x in ok_scene_tokens if x not in filtered_scene_list]
                                        # ↑ 把测试场景排除掉
ok_scene_tokens = filtered_ok_scene_tokens
```

**问题**：提取特征时需要全部 850 个 scene 的特征（包括 150 个测试场景），否则评测时测试场景的 .npy 文件不存在会导致 KeyError。这段过滤代码在训练编码器时是正确的（只用 700 scene），但在提取特征时必须禁用。

**修复方法**：注释掉第 104–120 行整个过滤块：

```python
# 修改后：
# ################# for 700 scenes only #################
# （全部注释掉，不进行过滤，加载全部 850 个 scene）
# seq_meta_path = ...
# filtered_scene_list = ...
# filtered_ok_scene_tokens = ...
# ok_scene_tokens = filtered_ok_scene_tokens
```

> 提示：如果不想修改 loader.py，也可以分两次运行——第一次正常运行提取 700 个 train scene 的特征，第二次修改过滤条件为 `seq_data[:900]`（取前 4200 条 train 序列过滤掉）来提取 150 个 test scene 的特征。但直接注释掉过滤块更简单。

#### 特征提取的 batch-size 与显存

【remote4b4dl】仓库的 `encoders/lidarclip/README.md` 指出：在 RTX 5090（32GB）上使用 ViT-L/14 训练时 `--batch-size` 不宜超过 32。特征提取阶段同理——如果 OOM，可减小 batch-size 或分批运行。特征提取本身比训练快得多（不计算梯度），850 个 scene 约 3.4 万帧，单卡数小时内可完成。

### Step 4：准备 Stage 1 数据（LiDAR-LLM）

从 LiDAR-LLM 官方仓库下载 **LiDAR-LLM-Nu-Caption**（162k QA，基于 nuScenes 静态单帧）。**官方仓库逻辑**（`datageneration/tools/build_stage1_from_lidarllm.py`）：

```json
[
  {
    "scene_id": "<sample_token，必须与 stage1_features_sample 下的 npy 文件名一致>",
    "scene_token": "<nuScenes scene_token>",
    "conversations": [
      {"from": "human", "value": "<video>\n问题文本"},
      {"from": "gpt",   "value": "答案文本"}
    ]
  },
  ...
]
```

官方做法：直接读 HF `Senqiao/LiDAR-LLM-Nu-Caption` 全量数据，按 `assets/sample_token_to_scene.json` + `assets/train_scene_tokens.json`（699 个训练 scene）过滤，**不做帧映射**，conversation id 就是 sample_token 本身；特征由 `extract_pc_features_sample_token.py` 每帧一个 `{sample_token}.npy` 另提。

本地复刻：用 `scene_metadata.json` 的 700 个 train scenes 近似官方 699 scenes（实测 0 泄漏、161,845 条全命中）。命令：

```bash
# 1) 生成 stage1_train.json（161,845 条，scene_id=sample_token）
python3 datageneration/tools/build_stage1_from_lidarllm.py \
    --llm_train /path/to/lidarllm_train.json \
    --sample_json /path/to/nuScenes/v1.0-trainval/sample.json \
    --scene_metadata /path/to/scene_metadata.json \
    --output ./mllm/b4dl_dataset/stage1_train.json

# 2) 提取特征（700 train scenes 全部关键帧，28,130 帧）
cd encoders/lidarclip
conda run -n wqlc python extract_pc_features_sample_token.py \
    --checkpoint ./lidarclip/checkpoint/vit_l_14.ckpt \
    --scene-metadata /root/autodl-tmp/wql/mmb4dl/dataset/nuScenes-B4DL/metadata/scene_metadata.json \
    --sample-json /root/autodl-tmp/Datasets/nuScenes/v1.0-trainval/sample.json \
    --data-path /root/autodl-tmp/Datasets/nuScenes \
    --save-dir ./b4dl/stage1_features_sample
```

要点：`scene_id` 必须是 sample_token（dataset.py 用它拼 `{feat_folder}/{scene_id}.npy`）；旧 frame_id 键控方案（`convert_lidarllm_to_stage1.py` + `stage1_features/`，95k 条）仅保留给旧 checkpoint；`stage1_val.json` 的 42,597 条全在 test scenes，按官方逻辑会被过滤，只作监控用。

### Step 5：Stage 1 训练 —— 3D LiDAR Understanding（只训 projector）

```bash
cd mllm    # 使用环境 B
bash run_stages.sh \
     --s1_data ./b4dl_dataset/stage1_lidarllm_mm.json \
     --s1_feat ./b4dl/stage1_features \
     --s2_data ./b4dl_dataset/stage2.json \
     --s2_feat ./b4dl/stage2_features \
     --model_name_or_path ./base_model/vicuna-v1-5-7b
```

或单独跑 Stage 1：`bash scripts/stage1.sh`。超参数（脚本实测值）：

| 参数 | Stage 1 | Stage 2 |
|---|---|---|
| 入口 | `deepspeed ... train_mem.py --deepspeed zero3.json` | 同左 |
| GPU | 单卡（脚本 `--include localhost:1`，可改） | 单卡（`localhost:0`） |
| 可训练部分 | 仅 `mm_projector`（`--tune_mm_mlp_adapter True`） | 仅 LoRA（`--lora_enable True`），冻结 projector |
| 对话模板 version | plain | v1 (vicuna) |
| Epochs | 1 | 2 |
| Batch | 16 × grad_accum 8 = 有效 128 | 8 × grad_accum 16 = 有效 128 |
| 学习率 | 1e-3，cosine，warmup 3% | 1e-4，同左 |
| LoRA | — | r=64, alpha=128, dropout=0.05, target=所有 Linear（除 lm_head） |
| max_length | 2048 | 2048 |
| 输出 | `checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin` | stage2 目录（LoRA 权重 + non_lora_trainables.bin） |

Stage 1 用 LiDAR-LLM 162k 数据跑 1 epoch ≈ 1270 steps，单卡数小时内完成。

### Step 6：Stage 2 训练 —— 4D LiDAR Understanding（只训 LoRA）

`bash scripts/stage2.sh`（run_stages.sh 会自动串联）。要点：

- 自动加载 Stage 1 的 `mm_projector.bin` 并冻结（`--freeze_mm_mlp_adapter True`）。
- 输入是整 scene 的 (num_frames, 768) 4D 特征，`<video>` 占位符处注入全部帧 token（论文 Figure 4 红框）。
- 论文提到此阶段给问题前缀 `<4DLiDAR>` 与 Metatoken `<meta>`，但**发布代码中未实现**（见 §五-6、7），复现最佳配置需自行补充。
- 148k 训练样本 × 2 epoch ÷ 128 ≈ 2320 steps，单卡数小时。

### Step 7：推理

仓库的 `vtimellm/inference.py` 仍是 VTimeLLM 视频版遗留（CLIP 抽帧），需要改写为：

1. 用 `vtimellm/model/builder.py` 的 `load_pretrained_model` 加载 base 模型 + stage1 projector + stage2 LoRA；
2. 视觉输入改为 `np.load(stage2_features/{scene_id}.npy)`；
3. 按测试集 JSON 逐条生成答案，保存 `{question, prediction, ground_truth, task}`。

若要复现论文最佳配置（Metatoken），需在问题文本前拼接（依据论文 Appendix C 与 Figure 6）：

```
<4DLiDAR>
<meta>
首帧元信息描述 ... 连接词 ... 末帧元信息描述
<video>
原始问题
```

元信息 = nuScenes ego_pose 计算的相邻帧相对方向/位置/速度/加速度的文本化（需要自己写一个从 nuScenes 提取 ego pose 并转文本的脚本，仓库未提供）。论文 Figure 6 给出了完整样例：输入问题为 *"What can be seen on the right side of the road from frame 10 to frame 22?"*，拼接后的输入形如 `<4DLiDAR> <meta> The metadata of the first frame is 'The ego vehicle is ...' and the metadata of the last frame is 'The ego vehicle is ...'`，其中元信息文本如 *"The ego vehicle is slightly ahead and to the right on flat ground … 5 meters per second … a left turn by about 11 degrees … acceleration of about 1 meters per second squared."*。数据来源是 nuScenes 每帧的 `calibrated_sensor` 与 `ego_pose` 记录（translation/rotation/timestamp），计算相对前一帧的位移、航向角变化（→速度/方向）、二阶差分（→加速度），再用模板转成自然语言；只对序列首帧和末帧生成，用连接词（论文图中红色高亮，如 "and the metadata of the last frame is"）拼接。

### Step 8：评测

原始仓库 `vtimellm/eval/` 是 VTimeLLM 原版（ActivityNet 的 DVC/grounding 评测），与论文六任务不匹配。有两种方案：

#### 方案 A【remote4b4dl】：使用修改版仓库自带的评测代码

remote4b4dl 仓库在 `mllm/evaluation/` 目录下已实现了完整的六任务评测代码：

| 文件 | 功能 |
|---|---|
| `evaluation/test_b4dl.py` | 端到端评测：加载模型+特征，逐条推理，输出预测 JSON，计算全部指标 |
| `evaluation/evaluate_model.py` | 指标库：Accuracy、mIoU、BLEU-4、METEOR、ROUGE-L、BERTScore、GPT Score（Table 9 prompt） |
| `evaluation/build_test_split.py` | 从生成的数据中按 nuScenes 官方 val split 提取测试集并格式化 |
| `evaluation/split_dataset.py` | 验证 train/test 场景数与论文 Table 2 对齐 |

评测命令：

```bash
cd mllm
# conda activate wqlc

# 完整评测（含 GPT Score，需 OPENAI_API_KEY）
python evaluation/test_b4dl.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2 \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --test_data ./b4dl_dataset/stage2_test.json \
    --output ./evaluation/predictions.json \
    --metrics_output ./evaluation/evaluation_results.json \
    --use_gpt --gpt_api_key $OPENAI_API_KEY

# 快速冒烟（只跑 10 条）
python evaluation/test_b4dl.py \
    --model_base ./base_model/vicuna-v1-5-7b \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage2 \
    --feat_folder ../encoders/lidarclip/b4dl/stage2_features \
    --test_data ./b4dl_dataset/stage2_test.json \
    --max_samples 10
```

`evaluate_model.py` 也可单独使用，对已有的预测结果计算指标：

```bash
python evaluation/evaluate_model.py \
    --predictions ./evaluation/predictions.json \
    --ground_truth ./b4dl_dataset/stage2_test.json
```

#### 方案 B：使用本方案附录 B 的自建评测代码

若使用原始 B4DL 仓库（无内置评测），可使用本方案附录 B 提供的完整 Python 脚本（`eval_all.py` 等），详见文档末尾。

#### 指标体系

| 任务 | 指标 | 实现 |
|---|---|---|
| Existence | Accuracy | 精确匹配（归一化：小写、去标点；答案为 Yes/No 或物体类别） |
| Binary QA | Accuracy | 同上（Yes/No） |
| Time Grounding | mIoU | 正则解析 `from frame (\d+) to frame (\d+)`，按帧区间算 IoU，解析失败记 0 |
| Description | B@4 / METEOR / ROUGE-L / BERTScore | `pycocoevalcap`（Bleu/Meteor/Rouge）+ `rouge-score` + `bert-score`（默认模型 F1） |
| Temporal Understanding | 同上 | 同上 |
| Comprehensive Reasoning | 同上 | 同上 |
| 全部 Complex 任务 | GPT Score | GPT-4o reference-free 打分，0–100，prompt 用论文 Table 9 原文 |

**汇总方式（论文 §5.1）**：Accuracy = Existence 与 Binary QA 的平均；mIoU = Time Grounding；B@4/ROUGE-L/METEOR/BERTScore/GPT Score = 三个 Complex 任务的平均。

GPT Score 的评测 prompt 已在论文 Table 9 给出全文（`Question: {q} GT: {ref} Answer: {pred} ... 0-100 分，只输出分数`），直接调用 OpenAI API 实现即可。

---

## 五、⚠️ 代码与论文的差异与坑位清单（复现成败关键）

以下是逐行审查发现的全部问题，按严重程度排序。标注【remote4b4dl 已修复】的项在修改版仓库中已解决，无需再处理。

1. **【致命】特征维度不匹配 768 vs 128**【remote4b4dl 已修复】。编码器（AttentionPool2d，embed_dim=CLIP ViT-L/14 的 768）输出 768 维，但 `mllm/vtimellm/model/vtimellm_arch.py:13` 是 `nn.Linear(128, hidden_size)`（768 版本被注释在上一行）。直接串联会在 stage1 加载或 forward 时维度报错。**修复**：把 `mm_projector` 改回 `nn.Linear(768, ...)`（与 dataset.py:400 的 `(100,768)` 占位一致，推断作者实际用 768）。同时检查 `dataset.py:407` 对 `(1,)` 形状特征的处理。

2. **【严重】评测代码缺失**【remote4b4dl 已修复】。论文的六任务指标体系与 GPT Score 均未发布，`eval/` 目录是 VTimeLLM 遗留。原始仓库需按 §四 Step 8 方案 B 自建评测脚本。remote4b4dl 仓库已在 `mllm/evaluation/` 下提供完整评测代码（`test_b4dl.py` + `evaluate_model.py` + `build_test_split.py`），可直接使用（见 §四 Step 8 方案 A）。

3. **【严重】Metatoken 未实现**。论文 §4.1/Appendix C 的核心模块（`<meta>` + ego 运动文本）在代码中完全不存在（全仓库 grep 无 `<meta>`；`<4DLiDAR>` 只出现在 `datageneration/utils.py` 一个未被调用的函数里）。复现 Table 4 最佳行（HA✓ + Metatoken✓）必须自己实现元数据文本化与拼接。

4. **【严重】数据生成脚本的 break bug**【remote4b4dl 已修复】：`generate_description.py:140` 的 `break` 使每次只处理 10 条序列。

5. **【严重】loader.py 测试场景过滤导致特征缺失**：`lidarclip/loader.py` 第 104–120 行过滤掉 150 个测试场景，提取特征时会导致测试场景无 .npy 文件，评测时 KeyError。**修复**：提取特征前注释掉整个过滤块（见 §四 Step 3 详解）。

6. **【严重】HF 数据集格式不匹配训练脚本**：HuggingFace `ccho4702/nuScenes-B4DL` 的 JSON 是 raw 格式（`question`/`answer` 字段），而 `stage2.sh` 需要的是 conversations 格式（`conversations` 数组 + `<video>` 前缀）。**修复**：使用 remote4b4dl 自带的 `stage2_conversations.json`，或自己转换格式（见 §2.2 格式示例）。

7. **【中】prompts.py 的 HA 分支 NameError**（变量自引用未赋值），复现 HA 消融需修复。

8. **【中】Time Grounding 数据生成代码缺失**（只有 5/6 个任务），且 `config.py` 的 `temporal_understanding` 与 `generate_dataset.py` 的 `temporal` 名称不一致。

9. **【中】硬编码路径**【remote4b4dl 已修复】：`loader.py:106` 原本硬编码 `/home/youngwoo.shin/...`，修改版已改为相对路径 `osp.join(osp.dirname(__file__), "..", "annotations", ...)`。但 `extract_description.py:401`、`extract_pc_features.py` 的占位 `--data-path`、`stage1.sh/stage2.sh` 中 `./lidarllm_only_dataset/...` 与 README 的 `./b4dl/...` 路径仍需统一。

10. **【中】样例数据格式陷阱**：`mllm/data/lidar-stage3_750.json` 用 `id` 字段，而 `dataset.py:404` 读 `source['scene_id']` → KeyError → 被 except 捕获后 `return random.choice(self)` **静默无限重采样**。训练数据务必带 `scene_id`，且建议把 dataset.py 的 except 分支改成 raise，先跑一遍确认所有 npy 都能找到，否则"训练在跑"实际是死循环。

11. **【低】依赖问题**：numpy 2.x 与 transformers 4.31 不兼容；requirements 重复 pin；缺 `wandb`/`torchvision`；flash-attn 需编译；`run_stages.sh` 中 COMMON_ARGS 排在 stage 参数之后会覆盖默认值（传 `--model_name_or_path` 时注意）。【remote4b4dl 已修复】仓库 requirements.txt 已清理重复 pin，但 numpy 版本仍需注意（用 `numpy<2`）。

12. **【低】Waymo 实验无代码**：datageneration 是 nuScenes 专用（6 路环视、前后分组），Waymo 需改成 5 路前向相机左右分组（论文 §5.3），属于可选的高级复现项。

---

## 六、对比模型复现（Table 3 需要）

| 模型 | 复现方式 | 工作量 |
|---|---|---|
| **B4DL-LiDARLLM** | 用同一套框架，但 stage2 只用 LiDAR-LLM 单帧数据（特征用 stage1 的单帧 npy，QA 用 LiDAR-LLM 数据集），即"只训 3D 不训 4D"的变体 | 低（改数据即可） |
| **VTimeLLM** | 克隆原版 VTimeLLM（github.com/huangb23/VTimeLLM），输入改为 nuScenes 前/后视图像序列渲染的"视频"（CLIP ViT-L/14 抽帧特征），在 B4DL 测试集上评测 | 中–高 |
| **B4DL (Ours†)** | 仅在 nuScenes 上训练，直接在 Waymo 1k 子集推理（需先完成 Waymo 数据生成与特征提取） | 高（可选） |

---

## 七、推荐执行路线与冒烟测试

### 7.1 冒烟测试（1–2 天，验证全链路连通）

1. 用 nuScenes-mini（10 scene）+ HF 数据集的一个小子集；
2. 跳过编码器训练，直接随机初始化或用小数据快训 1 epoch；
3. Stage1/Stage2 各跑几十步，确认 loss 下降、checkpoint 可加载、推理可出文本；
4. 用 10 条测试样本跑通自建评测脚本。

### 7.2 完整复现时间线（单卡 4090 估计）

| 阶段 | 内容 | 预计耗时 |
|---|---|---|
| 第 1 周 | 环境搭建（两套）、下载 nuScenes/Vicuna/CLIP、修 bug | 3–5 天 |
| 第 1–2 周 | 数据集：优先下载 HF 数据集；同时跑通 datageneration 小样本 | 1–3 天 |
| 第 2 周 | LiDAR-CLIP 编码器训练（20 epoch） | 1–3 天 |
| 第 2–3 周 | 特征提取（850 scene 全帧）+ Stage1 数据整理 | 0.5–1 天 |
| 第 3 周 | Stage1 + Stage2 训练 | 合计 <1 天（论文：全部实验 24h 内） |
| 第 3–4 周 | 推理 + 评测脚本实现 + GPT Score | 3–5 天 |
| 第 4 周 | 消融（HA/Metatoken）、数据规模实验、对比模型 | 3–5 天 |

### 7.3 验收标准

- 主指标落在论文值附近（合成数据与训练随机性下建议 **±5% 相对容差**）：Acc≈0.76、mIoU≈0.31、B@4≈0.095、ROUGE-L≈0.32、METEOR≈0.28、BERTScore≈0.90、GPT≈59.5；
- 相对关系成立：B4DL > VTimeLLM > B4DL-LiDARLLM（尤其 mIoU 与 GPT Score 差距应显著）；
- 消融趋势：HA✓+Metatoken✓ 最佳；去 Metatoken 后 mIoU 明显下降；去 HA 后 Complex 任务的 B@4/METEOR/ROUGE-L 明显下降；
- 数据规模趋势：10%→100% 单调上升并趋于饱和。

---

## 八、关键文件速查表

| 文件 | 作用 | 注意事项 |
|---|---|---|
| `datageneration/generate_description.py` | GPT-4o 生成前/后视描述 | 原始版删 L140 的 break【remote4b4dl 已修复】 |
| `datageneration/generate_dataset.py` | 描述→五任务 QA | 缺 time_grounding；任务名 temporal |
| `datageneration/prompts.py` | 全部 prompt（对应论文 Table 7/8） | HA 分支 NameError |
| `datageneration/data/metadata/*.json` | 官方 850 scene / 5100 sequence 划分 | 自带，直接用 |
| `encoders/lidarclip/train.py` | LiDAR-CLIP 训练 | 强制 wandb（可改 `--report_to none`） |
| `encoders/lidarclip/lidarclip/loader.py` | nuScenes 图像-点云对加载 | L104–120 过滤测试场景，提取特征前必须注释【关键】 |
| `encoders/lidarclip/extract_pc_features.py` | stage1/2 特征提取 | 必须传 --data-path/--checkpoint；会清空重建输出目录 |
| `encoders/lidarclip/extract_description.py` | 规则化运动 GT（HA） | 默认路径硬编码 |
| `mllm/scripts/stage1.sh` / `stage2.sh` | 两阶段训练 | 路径以脚本为准；stage1 用 stage2_features（非 stage1） |
| `mllm/vtimellm/model/vtimellm_arch.py` | projector + 特征注入 | 原始版 L13 维度 128→改 768【remote4b4dl 已修复】 |
| `mllm/vtimellm/train/dataset.py` | 数据加载（读 scene_id.npy） | except 静默重采样，建议改 raise |
| `mllm/evaluation/test_b4dl.py` | 【remote4b4dl 独有】端到端评测 | 加载模型+特征→推理→计算全部指标 |
| `mllm/evaluation/evaluate_model.py` | 【remote4b4dl 独有】指标库 | Accuracy/mIoU/BLEU-4/METEOR/ROUGE-L/BERTScore/GPT Score |
| `mllm/evaluation/build_test_split.py` | 【remote4b4dl 独有】测试集构建 | 从生成数据按 nuScenes val split 提取测试集 |
| `mllm/b4dl_dataset/stage2_conversations.json` | 【remote4b4dl 独有】训练用数据 | 68,695 条 conversations 格式（已含 `<video>` 前缀） |
| `mllm/scripts/stage3.sh` | 【remote4b4dl 独有】Stage 3 训练 | 二次 LoRA SFT，需先 merge Stage2 |
| `mllm/vtimellm/demo_gradio.py` | 【remote4b4dl 独有】Web Demo | 路径需传绝对值 |
| `mmb4dl/mmb4dl.md` | 论文全文（含 Table 9 GPT 评分 prompt、Table 7/8 数据生成 prompt） | 评测实现依据 |

---

## 九、风险与备选方案

| 风险 | 影响 | 备选方案 |
|---|---|---|
| GPT-4o API 成本高/不可用 | 无法自建数据集 | 直接用 HF 数据集（方案 A）；GPT Score 改用 gpt-4o-mini 并声明差异 |
| mmcv-full 1.3.9 + torch 1.9 环境装不上 | 编码器无法训练 | 尝试 Docker（sst/ 仓库自带 docker 目录）；或联系作者要训练好的 encoder checkpoint |
| 768/128 维度问题判断错误 | 训练报错 | 以"能加载 stage1/stage2 npy 并跑通 forward"为准统一维度 |
| nuScenes 下载慢 | 阻塞全流程 | 先用 v1.0-mini 冒烟；full 版走网盘镜像 |
| 评测实现有偏差 | 数值与论文对不上 | 严格用论文 §5.1 的聚合口径；GPT Score prompt 用 Table 9 原文 |

---

*本方案基于仓库代码逐行审查（含 git 历史对比）与论文全文交叉验证编写。执行前请先通读 §五 坑位清单。*

---

## 附录 A：数据集详细说明

### A.1 nuScenes v1.0-trainval（基础数据）

**下载**：访问 https://www.nuscenes.org/nuscenes#download ，注册账号后下载 **v1.0-trainval**（Full dataset, ~300GB 压缩包）。若只需冒烟测试，下载 **v1.0-mini**（~4GB）。

**目录结构**（解压后须为标准格式）：

```
/path/nuScenes/
├── samples/            # keyframe 数据（2Hz）
│   ├── LIDAR_TOP/      # *.pcd.bin（float32, 5列: x/y/z/intensity/ring）
│   ├── CAM_FRONT/      # *.jpg
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   └── CAM_BACK_RIGHT/
├── sweeps/             # 非 keyframe 中间帧（同上结构）
└── v1.0-trainval/      # 元数据 JSON
    ├── scene.json
    ├── sample.json
    ├── sample_data.json
    ├── ego_pose.json
    ├── calibrated_sensor.json
    └── ...
```

**验证脚本**：

```python
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot='/path/nuScenes', verbose=True)
print(f"Scenes: {len(nusc.scene)}, Samples: {len(nusc.sample)}")
# 预期：850 scenes, 34149 samples
```

### A.2 B4DL 数据集（178k QA，推荐直接下载）

**HuggingFace**：`ccho4702/nuScenes-B4DL`

```bash
pip install huggingface_hub
huggingface-cli download ccho4702/nuScenes-B4DL --repo-type dataset --local-dir ./b4dl_hf
```

> 注意：本机访问 huggingface.co 可能受阻，需代理或镜像站。若下载失败，可在能访问 HF 的机器上下载后拷贝。

**下载后需整理为以下格式**：

**stage2.json**（Stage 2 训练 + 测试用，六任务合并）：

```json
[
  {
    "scene_id": "101198145",
    "task": "existence",
    "conversations": [
      {"from": "human", "value": "<video>\nWas a pedestrian present in frame 004?"},
      {"from": "gpt", "value": "Yes."}
    ]
  },
  {
    "scene_id": "101198145",
    "task": "time_grounding",
    "conversations": [
      {"from": "human", "value": "<video>\nWhen did the ego vehicle change lanes?"},
      {"from": "gpt", "value": "from frame 003 to frame 008"}
    ]
  }
]
```

关键字段：
- `scene_id`：必须与 `stage2_features/{scene_id}.npy` 文件名一致（不含 `.npy`）
- `task`：六选一（`existence`, `binary`, `time_grounding`, `description`, `temporal_understanding`, `comprehensive`）
- 首条 human 消息的 value 必须以 `<video>\n` 开头（`dataset.py` 用 `<video>` 做特征注入占位符）

**测试集划分**：仓库 `datageneration/data/metadata/scene_metadata.json` 中最后 150 个 scene 为测试集（对应 900 个 sequence / ~30,145 条 QA）。将测试样本单独提取为 `stage2_test.json`。

### A.3 LiDAR-LLM Nu-Caption 数据集（Stage 1 训练用）

**来源**：LiDAR-LLM 官方仓库 https://github.com/senqiaoyang/LiDAR-LLM ，在其 README 中提供数据下载链接（通常是 Google Drive 或 HuggingFace）。

**用途**：Stage 1 训练 mm_projector，使用单帧 3D LiDAR 数据（非时序）。

**整理为 stage1.json 格式**：

```json
[
  {
    "scene_id": "<frame_id>",
    "conversations": [
      {"from": "human", "value": "<video>\nDescribe the 3D scene."},
      {"from": "gpt", "value": "There are 3 cars ahead and 2 pedestrians on the right..."}
    ]
  }
]
```

此处 `scene_id` 对应 `stage1_features/{frame_id}.npy`（单帧特征）。

### A.4 模型权重

| 权重 | 下载地址 | 大小 | 放置路径 |
|---|---|---|---|
| Vicuna-7b-v1.5 | https://huggingface.co/lmsys/vicuna-7b-v1.5 | ~13.5GB | `mllm/base_model/vicuna-v1-5-7b/` |
| CLIP ViT-L/14 | `clip.load("ViT-L/14")` 自动下载；或 https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be9f2c229759e0a5e1ec2d/ViT-L-14.pt | ~1.7GB | `~/.cache/clip/` 或 `encoders/lidarclip/pretrained/` |

### A.5 数据集统计（论文 §3.3）

| 划分 | Scenes | Sequences | QA 对数 |
|---|---|---|---|
| 训练集 | 700 | 4200 | ~148,255 |
| 测试集 | 150 | 900 | ~30,145 |
| 合计 | 850 | 5100 | ~178,400 |

各任务在测试集中的分布（用于分任务评测）：

| 任务 | 类型 | 测试 QA 数（约） | 指标 |
|---|---|---|---|
| Existence | Simple | ~5,000 | Accuracy |
| Binary QA | Simple | ~5,000 | Accuracy |
| Time Grounding | Simple | ~5,000 | mIoU |
| Description | Complex | ~5,000 | B@4, METEOR, ROUGE-L, BERTScore, GPT Score |
| Temporal Understanding | Complex | ~5,000 | 同上 |
| Comprehensive Reasoning | Complex | ~5,000 | 同上 |

---

## 附录 B：评测代码实现（完整可运行）

以下代码需使用**环境 B**（`b4dl_mllm`），依赖安装见 §3.2。

### B.0 公共工具与数据加载

```python
# eval_utils.py — 评测公共工具
import json
import re
import string
from collections import defaultdict

def normalize_text(s: str) -> str:
    """文本归一化：小写、去标点、去多余空白"""
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_predictions(pred_path: str) -> list:
    """
    加载推理结果 JSON，每条格式：
    {
        "task": "existence" | "binary" | "time_grounding" | "description" | "temporal_understanding" | "comprehensive",
        "question": "...",
        "prediction": "...",
        "ground_truth": "..."
    }
    """
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def group_by_task(data: list) -> dict:
    """按 task 字段分组"""
    groups = defaultdict(list)
    for item in data:
        groups[item["task"]].append(item)
    return dict(groups)
```

### B.1 Accuracy（Existence + Binary QA）

```python
# eval_accuracy.py
import re
from eval_utils import normalize_text, load_predictions, group_by_task

def exact_match(pred: str, gt: str) -> bool:
    """精确匹配（归一化后）"""
    return normalize_text(pred) == normalize_text(gt)

def evaluate_accuracy(items: list) -> float:
    """
    计算 Accuracy = 精确匹配数 / 总数
    适用于 Existence 和 Binary QA 任务
    """
    if not items:
        return 0.0
    correct = sum(1 for item in items if exact_match(item["prediction"], item["ground_truth"]))
    return correct / len(items)

def run_accuracy_eval(pred_path: str):
    data = load_predictions(pred_path)
    groups = group_by_task(data)

    results = {}
    for task_name in ["existence", "binary"]:
        if task_name in groups:
            acc = evaluate_accuracy(groups[task_name])
            results[task_name] = acc
            print(f"[{task_name}] Accuracy: {acc:.4f} ({sum(1 for i in groups[task_name] if exact_match(i['prediction'], i['ground_truth']))}/{len(groups[task_name])})")

    # 汇总：Accuracy = Existence 与 Binary QA 的平均
    acc_values = [v for k, v in results.items()]
    if acc_values:
        mean_acc = sum(acc_values) / len(acc_values)
        print(f"\n[汇总] Accuracy (mean of simple QA tasks): {mean_acc:.4f}")
        return mean_acc
    return 0.0

if __name__ == "__main__":
    import sys
    run_accuracy_eval(sys.argv[1])
```

### B.2 mIoU（Time Grounding）

```python
# eval_miou.py
import re
from eval_utils import load_predictions, group_by_task

def parse_frame_range(text: str):
    """
    从文本中解析帧区间 "from frame X to frame Y"
    返回 (start, end) 或 None
    """
    # 匹配多种格式
    patterns = [
        r"from\s+frame\s+(\d+)\s+to\s+frame\s+(\d+)",
        r"frame\s+(\d+)\s*(?:to|-|~)\s*frame\s*(\d+)",
        r"from\s+(\d+)\s+to\s+(\d+)",
        r"frame\s+(\d+)\s*(?:to|-|~)\s*(\d+)",
    ]
    text_lower = text.lower().strip()
    for pat in patterns:
        m = re.search(pat, text_lower)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None

def compute_iou(pred_range, gt_range) -> float:
    """计算两个帧区间的 IoU"""
    pred_start, pred_end = pred_range
    gt_start, gt_end = gt_range
    # 交集
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0, inter_end - inter_start + 1)
    # 并集
    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union = union_end - union_start + 1
    return intersection / union if union > 0 else 0.0

def evaluate_miou(items: list) -> float:
    """
    计算 Time Grounding 的 mIoU
    解析失败记 IoU=0
    """
    if not items:
        return 0.0
    iou_sum = 0.0
    valid = 0
    for item in items:
        gt_range = parse_frame_range(item["ground_truth"])
        pred_range = parse_frame_range(item["prediction"])
        if gt_range is None:
            continue  # GT 无法解析，跳过
        valid += 1
        if pred_range is None:
            iou_sum += 0.0  # 预测解析失败，IoU=0
        else:
            iou_sum += compute_iou(pred_range, gt_range)
    return iou_sum / valid if valid > 0 else 0.0

def run_miou_eval(pred_path: str):
    data = load_predictions(pred_path)
    groups = group_by_task(data)

    if "time_grounding" in groups:
        miou = evaluate_miou(groups["time_grounding"])
        print(f"[time_grounding] mIoU: {miou:.4f}")
        print(f"\n[汇总] mIoU: {miou:.4f}")
        return miou
    else:
        print("Warning: No time_grounding samples found.")
        return 0.0

if __name__ == "__main__":
    import sys
    run_miou_eval(sys.argv[1])
```

### B.3 BLEU-4 / METEOR / ROUGE-L（Complex 任务）

```python
# eval_nlg_metrics.py
"""
BLEU-4 / METEOR / ROUGE-L 评测
依赖：pip install pycocoevalcap nltk rouge-score
"""
import json
import sys
from collections import defaultdict

from eval_utils import load_predictions, group_by_task

# ── BLEU-4 ──
def compute_bleu4(items: list) -> float:
    """使用 pycocoevalcap 计算 BLEU-4"""
    from pycocoevalcap.bleu.bleu import Bleu

    gts = {}
    res = {}
    for i, item in enumerate(items):
        gts[i] = [item["ground_truth"]]
        res[i] = [item["prediction"]]

    scorer = Bleu(4)
    score, scores = scorer.compute_score(gts, res)
    return score[3]  # BLEU-4 是第 4 个元素（index 3）

# ── METEOR ──
def compute_meteor(items: list) -> float:
    """使用 pycocoevalcap 计算 METEOR"""
    from pycocoevalcap.meteor.meteor import Meteor

    gts = {}
    res = {}
    for i, item in enumerate(items):
        gts[i] = [item["ground_truth"]]
        res[i] = [item["prediction"]]

    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    return score

# ── ROUGE-L ──
def compute_rouge_l(items: list) -> float:
    """使用 rouge_score 计算 ROUGE-L F1"""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for item in items:
        result = scorer.score(item["ground_truth"], item["prediction"])
        scores.append(result["rougeL"].fmeasure)
    return sum(scores) / len(scores) if scores else 0.0

def run_nlg_eval(pred_path: str):
    data = load_predictions(pred_path)
    groups = group_by_task(data)

    complex_tasks = ["description", "temporal_understanding", "comprehensive"]
    all_scores = {"bleu4": [], "meteor": [], "rouge_l": []}

    for task_name in complex_tasks:
        if task_name not in groups:
            print(f"Warning: No {task_name} samples found, skipping.")
            continue
        items = groups[task_name]

        b4 = compute_bleu4(items)
        met = compute_meteor(items)
        rl = compute_rouge_l(items)

        print(f"[{task_name}] BLEU-4: {b4:.4f} | METEOR: {met:.4f} | ROUGE-L: {rl:.4f}")
        all_scores["bleu4"].append(b4)
        all_scores["meteor"].append(met)
        all_scores["rouge_l"].append(rl)

    # 汇总：三个 Complex 任务的平均
    if all_scores["bleu4"]:
        mean_b4 = sum(all_scores["bleu4"]) / len(all_scores["bleu4"])
        mean_met = sum(all_scores["meteor"]) / len(all_scores["meteor"])
        mean_rl = sum(all_scores["rouge_l"]) / len(all_scores["rouge_l"])
        print(f"\n[汇总] BLEU-4: {mean_b4:.4f} | METEOR: {mean_met:.4f} | ROUGE-L: {mean_rl:.4f}")
        return mean_b4, mean_met, mean_rl
    return 0.0, 0.0, 0.0

if __name__ == "__main__":
    run_nlg_eval(sys.argv[1])
```

### B.4 BERTScore（Complex 任务）

```python
# eval_bertscore.py
"""
BERTScore 评测
依赖：pip install bert-score
"""
import sys
from eval_utils import load_predictions, group_by_task

def compute_bertscore(items: list, model_type: str = "microsoft/deberta-xlarge-mnli") -> float:
    """
    计算 BERTScore F1
    默认使用 deberta-xlarge-mnli（BERTScore 论文推荐）
    若显存不足可换 "roberta-large"
    """
    from bert_score import score as bert_score_fn

    preds = [item["prediction"] for item in items]
    refs = [item["ground_truth"] for item in items]

    # bert_score 返回 (P, R, F1)，每个是 list
    P, R, F1 = bert_score_fn(
        preds, refs,
        model_type=model_type,
        lang="en",
        verbose=False,
        batch_size=64,
        num_workers=4
    )
    return F1.mean().item()

def run_bertscore_eval(pred_path: str, model_type: str = "microsoft/deberta-xlarge-mnli"):
    data = load_predictions(pred_path)
    groups = group_by_task(data)

    complex_tasks = ["description", "temporal_understanding", "comprehensive"]
    scores = []

    for task_name in complex_tasks:
        if task_name not in groups:
            print(f"Warning: No {task_name} samples found, skipping.")
            continue
        items = groups[task_name]
        bs = compute_bertscore(items, model_type)
        print(f"[{task_name}] BERTScore F1: {bs:.4f}")
        scores.append(bs)

    if scores:
        mean_bs = sum(scores) / len(scores)
        print(f"\n[汇总] BERTScore (mean of complex tasks): {mean_bs:.4f}")
        return mean_bs
    return 0.0

if __name__ == "__main__":
    run_bertscore_eval(sys.argv[1])
```

### B.5 GPT Score（Complex 任务，Reference-free 语义评分）

```python
# eval_gpt_score.py
"""
GPT Score 评测（0-100，使用论文 Table 9 的完整 prompt）
依赖：pip install openai
需要环境变量 OPENAI_API_KEY
"""
import json
import sys
import time
import os
from eval_utils import load_predictions, group_by_task

# ── 论文 Table 9 原文 prompt ──
GPT_SCORING_PROMPT = """You are an expert evaluator for semantic answer quality. You will be given a set of question-answer-ground truth (Q/A/GT) triplets. Your task is to evaluate how semantically close the given answer (A) is to the ground truth (GT), and how well it responds to the question (Q).

Focus only on:
- Whether the answer conveys the same or similar meaning as the ground truth.
- Whether the answer correctly and sufficiently addresses the question.
- Do not consider wording, phrasing, or grammar unless they change the meaning.

Score the answer from 0 to 100:
- 100: Fully aligned with GT in meaning, complete and relevant.
- 80-99: Mostly aligned with minor omissions or slight deviation.
- 60-79: Partially aligned; captures key ideas but misses or misrepresents some.
- 40-59: Limited relevance or meaning overlap with GT.
- 20-39: Barely related to the GT.
- 0-19: Meaningless or unrelated.

EXAMPLES:
Question: What changes occur to the car in front over the frames?
Answer: The car in front moves slightly forward from frame 000 to frame 008.
GT: The car in front moves slightly forward until it stops again from frame 0 to frame 8.
Score: 90

Question: From frame 004 to frame 006, how does the building structure in the back view affect the scene?
Answer: The building structure on the left side of the road in the back view remains constant, providing a stable reference point for the ego vehicle's navigation.
GT: The building structure remains fixed, providing a static backdrop in the scene without impacting the dynamics.
Score: 100

Question: How do the cars to the right change over time from frame 00 to frame 16?
Answer: The cars to the right remain stationary, indicating they are parked.
GT: They become slightly closer to the ego vehicle.
Score: 50

Question: What moving vehicles are visible in the ego vehicle's path by frame 026?
Answer: A car is moving away from the ego vehicle.
GT: There are no significant moving vehicles in the visible path.
Score: 10

INSTRUCTIONS
Question: {question}
GT: {ground_truth}
Answer: {prediction}

Please provide a score between 0 and 100 based on the quality of the prediction compared to the reference.
The score should reflect how well the prediction aligns with the reference in terms of semantic similarity and relevance to the question. Only provide the score without any additional text."""


def gpt_score_single(client, question: str, prediction: str, ground_truth: str,
                     model: str = "gpt-4o", max_retries: int = 3) -> int:
    """调用 GPT 对单条 QA 打分，返回 0-100 整数"""
    prompt = GPT_SCORING_PROMPT.format(
        question=question,
        prediction=prediction,
        ground_truth=ground_truth
    )
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            text = response.choices[0].message.content.strip()
            # 提取数字
            import re
            m = re.search(r"(\d+)", text)
            if m:
                score = int(m.group(1))
                return min(max(score, 0), 100)
            return 0
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  GPT scoring failed: {e}")
                return 0

def evaluate_gpt_score(items: list, model: str = "gpt-4o") -> float:
    """对一批样本计算 GPT Score 均值"""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    scores = []
    for i, item in enumerate(items):
        score = gpt_score_single(
            client,
            question=item["question"],
            prediction=item["prediction"],
            ground_truth=item["ground_truth"],
            model=model
        )
        scores.append(score)
        if (i + 1) % 100 == 0:
            print(f"  Scored {i+1}/{len(items)}, running mean: {sum(scores)/len(scores):.2f}")

    return sum(scores) / len(scores) if scores else 0.0

def run_gpt_score_eval(pred_path: str, model: str = "gpt-4o"):
    data = load_predictions(pred_path)
    groups = group_by_task(data)

    complex_tasks = ["description", "temporal_understanding", "comprehensive"]
    scores = []

    for task_name in complex_tasks:
        if task_name not in groups:
            print(f"Warning: No {task_name} samples found, skipping.")
            continue
        items = groups[task_name]
        print(f"Scoring [{task_name}] ({len(items)} samples) with {model}...")
        avg = evaluate_gpt_score(items, model)
        print(f"[{task_name}] GPT Score: {avg:.2f}")
        scores.append(avg)

    if scores:
        mean_score = sum(scores) / len(scores)
        print(f"\n[汇总] GPT Score (mean of complex tasks): {mean_score:.2f}")
        return mean_score
    return 0.0

if __name__ == "__main__":
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4o"
    run_gpt_score_eval(sys.argv[1], model)
```

### B.6 一键汇总评测脚本

```python
# eval_all.py
"""
B4DL 六任务一键评测
用法：python eval_all.py predictions.json [--gpt-model gpt-4o]

predictions.json 格式：
[
    {"task": "existence", "question": "...", "prediction": "...", "ground_truth": "..."},
    ...
]
"""
import json
import sys
import argparse

from eval_utils import load_predictions, group_by_task
from eval_accuracy import evaluate_accuracy
from eval_miou import evaluate_miou
from eval_nlg_metrics import compute_bleu4, compute_meteor, compute_rouge_l
from eval_bertscore import compute_bertscore
from eval_gpt_score import evaluate_gpt_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_path", type=str, help="推理结果 JSON 路径")
    parser.add_argument("--gpt-model", type=str, default="gpt-4o", help="GPT Score 用模型")
    parser.add_argument("--skip-gpt", action="store_true", help="跳过 GPT Score（省 API 费用）")
    parser.add_argument("--bertscore-model", type=str, default="microsoft/deberta-xlarge-mnli")
    args = parser.parse_args()

    data = load_predictions(args.pred_path)
    groups = group_by_task(data)

    print("=" * 60)
    print("B4DL Evaluation Report")
    print("=" * 60)

    results = {}

    # ── Simple Tasks ──
    # 1. Accuracy (Existence + Binary)
    acc_tasks = ["existence", "binary"]
    acc_vals = {}
    for t in acc_tasks:
        if t in groups:
            acc = evaluate_accuracy(groups[t])
            acc_vals[t] = acc
            print(f"  [{t}] Accuracy = {acc:.4f} (n={len(groups[t])})")
    mean_acc = sum(acc_vals.values()) / len(acc_vals) if acc_vals else 0.0
    results["Accuracy"] = mean_acc
    print(f"  ► Accuracy (mean) = {mean_acc:.4f}\n")

    # 2. mIoU (Time Grounding)
    if "time_grounding" in groups:
        miou = evaluate_miou(groups["time_grounding"])
        results["mIoU"] = miou
        print(f"  [time_grounding] mIoU = {miou:.4f} (n={len(groups['time_grounding'])})")
        print(f"  ► mIoU = {miou:.4f}\n")
    else:
        results["mIoU"] = 0.0
        print("  [time_grounding] No samples found.\n")

    # ── Complex Tasks ──
    complex_tasks = ["description", "temporal_understanding", "comprehensive"]
    metric_funcs = {
        "BLEU-4": compute_bleu4,
        "METEOR": compute_meteor,
        "ROUGE-L": compute_rouge_l,
    }

    for metric_name, metric_fn in metric_funcs.items():
        vals = []
        for t in complex_tasks:
            if t in groups:
                v = metric_fn(groups[t])
                vals.append(v)
                print(f"  [{t}] {metric_name} = {v:.4f}")
        mean_v = sum(vals) / len(vals) if vals else 0.0
        results[metric_name] = mean_v
        print(f"  ► {metric_name} (mean) = {mean_v:.4f}\n")

    # BERTScore
    print("Computing BERTScore (this may take a while on CPU)...")
    bs_vals = []
    for t in complex_tasks:
        if t in groups:
            v = compute_bertscore(groups[t], model_type=args.bertscore_model)
            bs_vals.append(v)
            print(f"  [{t}] BERTScore = {v:.4f}")
    mean_bs = sum(bs_vals) / len(bs_vals) if bs_vals else 0.0
    results["BERTScore"] = mean_bs
    print(f"  ► BERTScore (mean) = {mean_bs:.4f}\n")

    # GPT Score
    if not args.skip_gpt:
        print(f"Computing GPT Score with {args.gpt_model}...")
        gs_vals = []
        for t in complex_tasks:
            if t in groups:
                v = evaluate_gpt_score(groups[t], model=args.gpt_model)
                gs_vals.append(v)
                print(f"  [{t}] GPT Score = {v:.2f}")
        mean_gs = sum(gs_vals) / len(gs_vals) if gs_vals else 0.0
        results["GPT Score"] = mean_gs
        print(f"  ► GPT Score (mean) = {mean_gs:.2f}\n")
    else:
        print("  [GPT Score] Skipped (--skip-gpt)\n")

    # ── Summary Table ──
    print("=" * 60)
    print("SUMMARY (Table 3 format)")
    print("=" * 60)
    header = f"{'Accuracy':>10} {'mIoU':>8} {'B@4':>8} {'ROUGE-L':>10} {'METEOR':>8} {'BERTScore':>10} {'GPT Score':>10}"
    print(header)
    gpt_str = f"{results.get('GPT Score', 0):.3f}" if "GPT Score" in results else "N/A"
    print(f"{results['Accuracy']:>10.3f} {results['mIoU']:>8.3f} {results['BLEU-4']:>8.3f} "
          f"{results['ROUGE-L']:>10.3f} {results['METEOR']:>8.3f} {results['BERTScore']:>10.3f} {gpt_str:>10}")

    # 保存结果
    out_path = args.pred_path.replace(".json", "_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
```

### B.7 运行示例

```bash
# 完整评测（含 GPT Score，需 OPENAI_API_KEY）
export OPENAI_API_KEY="sk-..."
python eval_all.py ./outputs/predictions.json --gpt-model gpt-4o

# 冒烟评测（跳过 GPT Score，省费用）
python eval_all.py ./outputs/predictions.json --skip-gpt

# 单独运行某指标
python eval_accuracy.py ./outputs/predictions.json
python eval_miou.py ./outputs/predictions.json
python eval_nlg_metrics.py ./outputs/predictions.json
python eval_bertscore.py ./outputs/predictions.json
python eval_gpt_score.py ./outputs/predictions.json gpt-4o-mini   # 用 mini 模型降本
```

### B.8 评测依赖安装清单

```bash
# 在环境 B (b4dl_mllm) 中
pip install pycocoevalcap==1.2 nltk rouge-score bert-score==0.3.13 openai

# METEOR 需要 nltk 数据
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')"

# BERTScore 首次运行会自动下载模型（deberta-xlarge-mnli ~1.5GB）
# 若显存不足，可换用较小模型：
# python eval_all.py predictions.json --bertscore-model roberta-large
```

### B.9 评测指标与论文 Table 3 的对应关系

| 论文指标 | 计算方式 | 涉及任务 | 聚合方式 |
|---|---|---|---|
| Accuracy | 精确匹配（归一化后） | Existence + Binary QA | 两任务 Accuracy 的算术平均 |
| mIoU | 帧区间 IoU（解析失败=0） | Time Grounding | 直接取该任务 mIoU |
| BLEU-4 | pycocoevalcap Bleu(4) | Description + Temporal + Comprehensive | 三任务 BLEU-4 的算术平均 |
| METEOR | pycocoevalcap Meteor | 同上 | 三任务 METEOR 的算术平均 |
| ROUGE-L | rouge-score rougeLsu F1 | 同上 | 三任务 ROUGE-L 的算术平均 |
| BERTScore | bert-score F1 (deberta-xlarge) | 同上 | 三任务 BERTScore 的算术平均 |
| GPT Score | GPT-4o 打分 0-100（Table 9 prompt） | 同上 | 三任务 GPT Score 的算术平均 |

**Table 3 目标值**：

| Accuracy↑ | mIoU↑ | B@4↑ | ROUGE-L↑ | METEOR↑ | BERTScore↑ | GPT Score↑ |
|---|---|---|---|---|---|---|
| 0.762 | 0.311 | 0.095 | 0.322 | 0.275 | 0.897 | 59.513 |
