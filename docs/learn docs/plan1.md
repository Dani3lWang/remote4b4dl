# B4DL / LiDAR-LLM 后续工作执行计划

> 本计划根据 `progress_report.md`、`learning_notes.md`、`plan_3_analysis.md` 以及项目实际代码整理而成。请按阶段顺序执行，每个任务完成后在复选框中打勾，并填写验收记录。

---

## 使用说明

- `- [ ]` 表示待完成；`- [x]` 表示已完成。
- **必须研读的代码片段**已经列在每一节中，建议先读代码再执行动作。
- 每一阶段末尾有 **验收标准**，未通过时不要进入下一阶段。

---

# 第一阶段：清理代码层面的“阻塞问题”

## 任务 1.1：确认并修复 `mm_projector` 维度不匹配

### 问题描述
`mllm/vtimellm/model/vtimellm_arch.py` 中 `mm_projector` 输入维度写为 128，但 SST + AttentionPool 实际输出 768 维。

### 必须研读的代码

#### 1.1.1 `mllm/vtimellm/model/vtimellm_arch.py` 第 8–20 行
```python
class VTimeLLMMetaModel:

    def initialize_vision_modules(self, model_args):
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        if not hasattr(self, 'mm_projector'):
            # self.mm_projector = nn.Linear(768, self.config.hidden_size)
            self.mm_projector = nn.Linear(128, self.config.hidden_size)
```
**思考**：为什么实际使用的是 128 而不是 768？训练脚本能否直接跑通？

#### 1.1.2 `encoders/lidarclip/extract_pc_features.py` 第 178–189 行
```python
clip_model, clip_preprocess = clip.load("ViT-L/14")  # output_dim = 768
lidar_encoder = LidarEncoderSST(
    "lidarclip/model/sst_encoder_only_config.py",
    clip_model.visual.output_dim  # = 768
)

# 逐帧提取特征
for batch in tqdm(loader):
    _, point_clouds, pc_path = batch[:3]
    lidar_features, _ = model.lidar_encoder(point_clouds)
    for lidar_feat, lidar_path in zip(lidar_features, pc_path):
        lidar_dict[lidar_path] = lidar_feat.unsqueeze(0)  # [1, 768]
```
**思考**：Stage2 保存的特征文件形状是否为 `[N_frames, 768]`？

#### 1.1.3 `encoders/lidarclip/lidarclip/model/sst.py` 第 117–131 行
```python
class LidarEncoderSST(nn.Module):
    def __init__(self, sst_config_path, clip_embedding_dim=512):
        super().__init__()
        self._sst = build_sst(sst_config_path)
        self._pooler = AttentionPool2d(
            spacial_dim=sst_model_conf["backbone"]["output_shape"][0],
            embed_dim=clip_embedding_dim,
            num_heads=8,
            input_dim=sst_model_conf["backbone"]["conv_out_channel"],
        )

    def forward(self, point_cloud, no_pooling=False, return_attention=False):
        lidar_features = self._sst.extract_feat(point_cloud, None)[0]  # bs, d, h, w
        pooled_feature, attn_weights = self._pooler(lidar_features, no_pooling, return_attention)
        return pooled_feature, attn_weights
```
**思考**：`AttentionPool2d` 输出维度由什么决定？是否为 768？

#### 1.1.4 `encoders/lidarclip/lidarclip/model/attention_pool.py` 第 143–165 行
```python
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim, embed_dim, num_heads, input_dim, output_dim=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.num_heads = num_heads

    def forward(self, x, no_pooling=False, return_attention=False):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = self.in_proj(x)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        query = x[0:1]
        x, weights = F.multi_head_attention_forward(
            query=query, key=x[1:], value=x[1:], ...
        )
        return x[0], weights
```
**思考**：`output_dim` 默认等于 `embed_dim`，这里 `embed_dim` 由 CLIP ViT-L/14 决定为 768。

### 执行动作
- [ ] 步骤 1：检查任意一个 Stage2 `.npy` 特征文件，确认形状为 `[N_frames, 768]`。
- [ ] 步骤 2：直接运行 `mllm/scripts/stage1.sh`，观察是否因维度不匹配报错。
- [ ] 步骤 3（分支 A：直接修复）：将 `mllm/vtimellm/model/vtimellm_arch.py` 第 13 行改为 `nn.Linear(768, self.config.hidden_size)`，重新训练 Stage1。
- [ ] 步骤 3（分支 B：保留 128 维）：在特征提取或 `mm_projector` 前增加 `nn.Linear(768, 128)`，并重新生成 Stage1/Stage2 特征。

### 验收标准
- [ ] Stage1 训练能够正常启动；
- [ ] 前 100 个 step 的 loss 呈现下降趋势；
- [ ] 没有 `mat1 and mat2 shapes cannot be multiplied` 类的维度错误。

### 执行记录
- 发现的问题：____________________
- 选择的方案（A/B）：____________________
- 是否通过验收：____________________

---

## 任务 1.2：梳理 LiDAR-LLM 三阶段学习与当前代码的对应关系

### 问题描述
项目根目录 `model.py` 只有 `pretrain` / `finetune` 两个 phase，而 `mllm/vtimellm/train/train.py` 有 `training_stage=1/2/3` 的完整三阶段逻辑。需要把这两者对应起来。

### 必须研读的代码

#### 1.2.1 `model.py` 第 101–129 行
```python
def get_trainable_params(self, phase='finetune'):
    for name, para in self.named_parameters():
        para.requires_grad = False

    if phase == 'finetune':
        for name, para in self.llama.named_parameters():
            if 'norm' in name:
                para.data = para.data.float()
                para.requires_grad = True
            if 'bias' in name:
                para.data = para.data.float()
                para.requires_grad = True
            if 'lora' in name:
                para.data = para.data.float()
                para.requires_grad = True

    elif phase == 'pretrain':
        train_param_name = ['Qformer', 'angle_pos_embd', 'query_tokens', \
                            'vision_proj', 'vision_proj_norm', 'bev', \
                            'gate', 'visual_query', 'visual_blocks', \
                             'adapter_query']
        for name, para in self.named_parameters():
            for train_name in train_param_name:
                if train_name in name:
                    para.data = para.data.float()
                    para.requires_grad = True
```
**思考**：`pretrain` 开放了 Qformer、vision_proj、adapter_query 等；`finetune` 开放了 LLaMA 的 norm/bias/lora。这与论文三阶段如何对应？

#### 1.2.2 `model.py` 第 133–164 行
```python
def forward_lidar(self, bev_feats, index=None):
    bev_feats = self.bev_conv1(bev_feats) 
    bev_feats = self.add_angle_embedding_optimized(bev_feats.float().permute(0,2,3,1))
    bev_feats = self.bev_proj_norm(self.bev_proj(bev_feats))
    bev_feats = bev_feats.float().reshape(bev_feats.shape[0], -1, bev_feats.shape[-1])
    bev_atts = torch.ones(bev_feats.size()[:-1], dtype=torch.long).to(bev_feats.device)
    query_tokens = self.query_tokens.expand(bev_atts.shape[0], -1, -1)
    condition = (index != -1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    selected_angle_embd = torch.where(condition, self.angle_pos_embd[index],
                              torch.mean(self.angle_pos_embd, dim=0, keepdim=True))
    selected_angle_embd = selected_angle_embd.squeeze(1).squeeze(1).expand_as(query_tokens)
    query_tokens = query_tokens + selected_angle_embd
    query_output = self.Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=bev_feats,
        encoder_attention_mask=bev_atts,
        return_dict=True,
    )
    bev_feat_former = torch.cat([torch.mean(query_output.last_hidden_state, dim=1, keepdim = True) , query_output.last_hidden_state], dim=1)
    bev_feat_former = F.normalize(
        self.vision_proj(bev_feat_former), dim=-1
    )
    return bev_feat_former
```
**思考**：Qformer 输出 576 个 token，加上 mean pool 变成 577 个。这与 BEV 分辨率 180×180 是什么关系？

#### 1.2.3 `mllm/vtimellm/train/train.py` 第 268–297 行
```python
if training_args.lora_enable:
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    if training_args.bits == 16:
        ...
    if training_args.training_stage == 3:
        model.get_model().initialize_vision_modules(model_args)
        model = load_lora(model, model_args.stage2_path)
        rank0_print('Merging LoRA weights...')
        model = model.merge_and_unload()
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
    else:
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
```
**思考**：Stage 3 为什么要先 merge Stage 2 的 LoRA，再添加新的 LoRA？

#### 1.2.4 `mllm/scripts/stage1.sh` 与 `mllm/scripts/stage2.sh`
```bash
# stage1.sh 关键参数
--tune_mm_mlp_adapter True
--data_path ./lidarllm_only_dataset/stage1_lidarllm_mm.json
--feat_folder ./lidarclip/stage1_features

# stage2.sh 关键参数
--lora_enable True
--data_path ./b4dl_dataset/stage2.json
--feat_folder ./lidarclip/stage2_features
--pretrain_mm_mlp_adapter ./checkpoints/.../mm_projector.bin
--num_train_epochs 2
```
**思考**：当前 B4DL 实现是 Stage1（对齐）+ Stage2（时序+SFT），缺少独立的 Stage3 SFT。这是否导致 QA 准确率不够高？

### 执行动作
- [ ] 步骤 1：绘制 LiDAR-LLM 三阶段与 `model.py` / `train.py` 的对应关系表。
- [ ] 步骤 2：确认 `model.py` 中 `pretrain` 对应论文哪一阶段，`finetune` 对应哪一阶段。
- [ ] 步骤 3：确认 B4DL 当前缺少 Stage3 是否是设计选择，还是实现遗漏。

### 阶段对应表（填写）

| LiDAR-LLM 论文阶段 | 训练目标 | `model.py` phase | B4DL 当前实现 | 更新哪些参数 |
|---|---|---|---|---|
| Stage 1 | 单帧特征对齐 | `pretrain` | `stage1.sh`（tune_mm_mlp_adapter） | mm_projector |
| Stage 2 | 多帧时序理解 | ? | `stage2.sh`（LoRA + 多帧数据） | LoRA |
| Stage 3 | 指令微调 SFT | `finetune` | 缺失 | LoRA / LLaMA norm+bias |

### 验收标准
- [ ] 能清楚解释每个阶段训练哪些参数；
- [ ] 能解释 Stage 3 为什么要 merge 旧 LoRA 再添加新 LoRA；
- [ ] 明确 B4DL 是否需要补齐 Stage3。

### 执行记录
- 对应关系结论：____________________
- 是否需要补 Stage3：____________________

---

# 第二阶段：诊断 B4DL 时序理解不足的根因

## 任务 2.1：用控制变量法排查根因

### 假设
1. QA 数据质量不足；
2. 训练轮次不够；
3. 架构本身时序建模能力有限。

### 必须研读的代码

#### 2.1.1 `mllm/vtimellm/train/dataset.py` 第 385–398 行
```python
if 'meta' in source:
    def convert(duration, x):
        x = x / duration * 100
        x = str(min(round(x), 99))
        if len(x) == 1:
            x = "0" + x
        return x

    replace_set = []
    for k, v in source['meta']['token'].items():
        replace_set.append((k, convert(source['meta']['duration'], v)))
    for l in range(len(source['conversations'])):
        for x1, x2 in replace_set:
            source['conversations'][l]['value'] = source['conversations'][l]['value'].replace(x1, x2)
```
**思考**：时间戳被压缩成两位数字符串，是否丢失了太多时序信息？

#### 2.1.2 `datageneration/generate_dataset.py` 第 147–181 行
```python
def generate_temporal_understanding_dataset(self, front_description, back_description, gt_description, start_index, end_index):
    client = OpenAI(api_key=self.cfg.API_KEY)
    content = self.prompts.generate_temporal_understanding_dataset_prompt(...)
    PROMPT_MESSAGES = [
        {
            "role": "system",
            "content": [{"type":"text", "text": "You are a helpful assistant that makes simple QnA pairs about the entire scene using the description of front and back parts of the ego vehicle."}],
        },
        {"role": "user", "content": [{"type": "text", "text": content}]},
    ]
    ...
    return result.choices[0].message.content
```
**思考**：生成的 temporal QA 是否过度依赖 front/back 描述，而不是真正需要多帧推理？

#### 2.1.3 `mllm/scripts/stage2.sh` 第 18 行
```bash
--num_train_epochs 2
```
**思考**：2 个 epoch 是否足够？loss 曲线在训练结束时是否仍在下降？

### 执行动作
- [ ] 步骤 1（数据质量）：从 `b4dl_dataset/stage2.json` 中随机抽取 50 条 `temporal` 类样本，人工判断问题是否需要多帧推理才能回答。
- [ ] 步骤 2（训练轮次）：保持数据和模型不变，分别跑 `num_train_epochs=2, 3, 4` 三组实验，记录 temporal QA 准确率。
- [ ] 步骤 3（架构能力）：构建“简单时序”与“复杂时序”测试集，评估当前模型在不同难度下的表现。

### 验收标准
- [ ] 明确根因是数据质量、训练轮次、架构限制中的哪一个或哪几个；
- [ ] 有量化数据支持（例如 temporal QA 准确率随 epoch 的变化曲线）。

### 执行记录
- 数据质量抽样结果：____________________
- 不同 epoch 的 temporal QA 准确率：____________________
- 根因结论：____________________

---

## 任务 2.2：调研 VTimeLLM 的细粒度时序建模

### 目标
评估是否可以引入更细粒度的时间建模来改善 B4DL。

### 执行动作
- [ ] 步骤 1：阅读 VTimeLLM 原论文/代码，记录其时序建模方式（时间戳嵌入、事件边界、时间定位头等）。
- [ ] 步骤 2：对比 `dataset.py` 第 385–398 行的时间戳转换逻辑，列出 3 个可低成本引入的改进点。
- [ ] 步骤 3：评估每个改进点的实现复杂度和预期收益。

### 可引入的改进点（填写）

| 改进点 | 实现位置 | 复杂度 | 预期收益 |
|---|---|---|---|
| 例：增加绝对时间嵌入 | `vtimellm_arch.py` | 低 | 让模型感知帧间绝对间隔 |
| | | | |
| | | | |

### 验收标准
- [ ] 至少列出 2 种可引入 VTimeLLM 的时序建模方式；
- [ ] 对每种方式给出实现复杂度和预期收益的初步判断。

---

# 第三阶段：数据生成优化（若根因包含数据质量）

## 任务 3.1：增强 temporal QA 的多样性和细粒度

### 必须研读的代码

#### 3.1.1 `datageneration/prompts.py`
找到 `generate_temporal_understanding_dataset_prompt` 函数，重点看：
- prompt 是否要求生成“运动变化类”问题；
- prompt 是否要求生成“多物体交互时序类”问题；
- prompt 是否显式要求利用时间范围。

#### 3.1.2 `datageneration/config.py`
确认 `TASK`、`START_INDEX`、`END_INDEX` 等配置项，了解如何控制生成范围。

### 执行动作
- [ ] 步骤 1：修改 `prompts.py` 中的 temporal prompt，明确要求生成：
  - 运动方向变化类问题；
  - 多物体出现时序类问题；
  - 显式时间范围类问题（如“在第 03 帧到第 07 帧之间”）。
- [ ] 步骤 2：在 `generate_dataset.py` 中增加后处理过滤：删除答案可直接从单帧描述推断出的 QA。
- [ ] 步骤 3：生成新的数据集 `stage2_temporal_enhanced.json`。

### 验收标准
- [ ] 新数据集中 temporal QA 数量不少于原版的 80%；
- [ ] 人工抽检 30 条，至少 70% 需要多帧推理才能回答。

### 执行记录
- 修改后的 prompt 要点：____________________
- 新数据集路径：____________________
- 抽检合格率：____________________

---

# 第四阶段：VAT 空间感知机制移植

## 任务 4.1：提取 LiDAR-LLM 的 VAT 核心逻辑

### 必须研读的代码

#### 4.1.1 `model.py` 第 58–69 行
```python
self.VIEW_RANGE = [
        [[0,35], [325,360]],    # FRONT
        [[270,340]],            # FRONT_LEFT
        [[20,90]],              # FRONT_RIGHT
        [[125,235]],            # BACK
        [[75,145]],             # BACK_LEFT
        [[215,285]]            # BACK_RIGHT
    ]
self.view_masks = self.generate_angles(C=768, H=180,W=180) 
self.angle_pos_embd = nn.Parameter(
    torch.ones((6, 1, 1, 1, 768))     
    )
```
**思考**：6 个视角的 FOV 划分是否适用于 nuScenes 的 6 相机布局？

#### 4.1.2 `model.py` 第 166–193 行
```python
def find_angle(self, angle, angle_ranges):
    mask = torch.zeros_like(angle, dtype=torch.bool).to(angle.device)
    for range in angle_ranges:
        mask = mask | (angle >= range[0]) & (angle <= range[1])
    return mask

def generate_angles(self, C=768, H=60, W=60):
    x_coords = torch.arange(-W // 2, W // 2)
    y_coords = torch.arange(-H // 2, H // 2)
    xx, yy = torch.meshgrid(x_coords, y_coords)
    angles = torch.atan2(xx, yy) * 180 / 3.14159265
    angles = (angles + 360) % 360
    view_masks = torch.zeros((6,H,W,C), dtype=torch.bool)
    for i in range(len(self.VIEW_RANGE)):
        range_ = self.VIEW_RANGE[i]
        mask = self.find_angle(angles, range_)
        mask = mask.unsqueeze(-1).repeat(1,1,C)
        view_masks[i] = mask
    return view_masks

def add_angle_embedding_optimized(self, bev_feat):
    B, H, W, C = bev_feat.shape
    view_masks_expanded = self.view_masks.unsqueeze(1).repeat(1, B, 1, 1, 1).to(bev_feat.device)
    angle_pos_embd_expanded = self.angle_pos_embd.expand(-1, B, H, W, C)
    pos_embd_aggregated = view_masks_expanded * angle_pos_embd_expanded 
    bev_feat = bev_feat + pos_embd_aggregated.sum(dim=0)
    return bev_feat
```
**思考**：`generate_angles` 中默认 `H=60, W=60`，但初始化时传的是 `H=180, W=180`，这是为什么？

### 执行动作
- [ ] 步骤 1：将 `model.py` 中 VAT 相关代码抽取到新文件 `mllm/vtimellm/model/angle_embedding.py`。
- [ ] 步骤 2：构造一个假输入 `torch.randn(2, 768, 180, 180)`，单独测试 VAT 的 forward，确认输出形状。

### 验收标准
- [ ] VAT 模块能独立运行；
- [ ] 输出形状符合预期。

---

## 任务 4.2：把 VAT 接到 B4DL 的 SST 特征上

### 必须研读的代码

#### 4.2.1 `encoders/lidarclip/lidarclip/model/sst.py` 第 128–131 行
```python
def forward(self, point_cloud, no_pooling=False, return_attention=False):
    lidar_features = self._sst.extract_feat(point_cloud, None)[0]  # bs, d, h, w
    pooled_feature, attn_weights = self._pooler(lidar_features, no_pooling, return_attention)
    return pooled_feature, attn_weights
```
**思考**：`no_pooling=True` 时能否输出二维 BEV 特征图？

#### 4.2.2 `encoders/lidarclip/extract_pc_features.py` 第 185–201 行
```python
for batch in tqdm(loader):
    _, point_clouds, pc_path = batch[:3]
    lidar_features, _ = model.lidar_encoder(point_clouds)
    for lidar_feat, lidar_path in zip(lidar_features, pc_path):
        lidar_dict[lidar_path] = lidar_feat.unsqueeze(0)  # [1, 768]

# Stage 2/3: 按 scene 拼接多帧
for d in tqdm(token_data):
    scene_id = d["scene_id"]
    scene_length = d["num_frames"]
    frames = d["paths"]["PATH_LIDAR_TOP"]
    feature_list = []
    for i in range(scene_length):
        frame_key = "PATH_{:03d}".format(i)
        frame_feature = lidar_dict[frames[frame_key]]  # [1, 768]
        feature_list.append(frame_feature)
    concat_lidar_feature = torch.cat(feature_list, dim=0)  # [N_frames, 768]
```
**思考**：当前流程把每帧压缩成 1 个 768 维向量，VAT 需要二维特征，如何衔接？

### 可选方案

| 方案 | 描述 | 优点 | 缺点 |
|---|---|---|---|
| A | 每帧保留 BEV 特征图，先加 VAT，再池化成序列 token | 空间感知在池化前注入 | token 数增加，显存压力大 |
| B | 多帧 BEV 在时序上聚合后统一加 VAT | 时空统一建模 | 实现复杂，时序对齐困难 |
| C | 在现有 `[N_frames, 768]` 序列上为每一帧额外学习一个“视角嵌入” | 改动最小 | 不是真正的 BEV 空间感知 |

### 执行动作
- [ ] 步骤 1：选择方案 A/B/C（建议先做 C 作为快速验证，再做 A）。
- [ ] 步骤 2：修改 `extract_pc_features.py` 或 `vtimellm_arch.py`，接入 VAT。
- [ ] 步骤 3：如果 token 数量变化，同步调整 `model_max_length` 和 batch size。

### 验收标准
- [ ] 训练能正常启动；
- [ ] 显存占用在可接受范围内。

### 执行记录
- 选择的方案：____________________
- 修改的文件：____________________
- 显存变化：____________________

---

# 第五阶段：对比实验

## 任务 5.1：定义实验组

| 实验组 | 模型 | 数据 | 目的 |
|---|---|---|---|
| A | 修复 mm_projector 后的 B4DL | 原版 `stage2.json` | 新 baseline |
| B | A + epoch=3/4 | 原版数据 | 验证训练轮次 |
| C | A + 增强 temporal QA | `stage2_temporal_enhanced.json` | 验证数据质量 |
| D | A + VAT 空间感知 | 原版或增强数据 | 验证空间感知增益 |
| E | A + VAT + 增强 temporal QA | 增强数据 | 验证时空双重感知 |

### 执行动作
- [ ] 步骤 1：确定每组实验的超参数（学习率、batch size、LoRA rank 等）保持一致。
- [ ] 步骤 2：为每组实验创建独立的输出目录，例如 `./checkpoints/exp_a_baseline`。

---

## 任务 5.2：拆分评估指标

### 必须研读的代码

#### 5.2.1 `mllm/vtimellm/eval/eval.py`
了解当前评估脚本的输入输出格式，以及它如何计算答案准确率。

### 执行动作
- [ ] 步骤 1：修改 `eval.py` 或新增脚本，按 `existence / description / temporal / comprehensive / binary` 分别统计准确率。
- [ ] 步骤 2：进一步把 `temporal` 拆分为“简单时序”和“复杂时序”。

### 验收标准
- [ ] 能输出每类任务的独立准确率；
- [ ] temporal 类能区分简单/复杂。

---

## 任务 5.3：跑实验并记录结果

### 执行动作
- [ ] 步骤 1：依次跑完实验组 A–E。
- [ ] 步骤 2：用 wandb 或表格记录每组实验的 loss 曲线和最终准确率。
- [ ] 步骤 3：把结果整理到 `progress_report.md` 或新建 `experiments/xx.md`。

### 实验结果记录表（填写）

| 实验组 | 数据 | 训练 epoch | temporal 简单 | temporal 复杂 | 综合 | 备注 |
|---|---|---|---|---|---|---|
| A | 原版 | 2 | | | | |
| B | 原版 | 3 | | | | |
| B' | 原版 | 4 | | | | |
| C | 增强 | 2 | | | | |
| D | 原版+VAT | 2 | | | | |
| E | 增强+VAT | 2 | | | | |

---

# 第六阶段：总结与下一步

### 执行动作
- [ ] 步骤 1：根据实验结果回答以下问题：
  - VAT 是否显著提升了 temporal QA？
  - 数据增强是否比改架构更有效？
  - B4DL 是否需要补齐 Stage3 SFT？
- [ ] 步骤 2：更新 `learning_notes.md` 和 `progress_report.md`。
- [ ] 步骤 3：确定下一步方向（继续优化数据、深入改架构、或写论文/报告）。

### 结论记录
- VAT 对 temporal QA 的效果：____________________
- 数据增强的效果：____________________
- 是否需要 Stage3：____________________
- 下一步方向：____________________

---

# 附录：关键文件索引

| 文件 | 作用 |
|---|---|
| `mllm/vtimellm/model/vtimellm_arch.py` | mm_projector、视觉 token 插入 |
| `mllm/vtimellm/train/train.py` | 三阶段训练逻辑 |
| `mllm/vtimellm/train/dataset.py` | 数据加载、时间戳转换 |
| `mllm/scripts/stage1.sh` | Stage1 训练脚本 |
| `mllm/scripts/stage2.sh` | Stage2 训练脚本 |
| `encoders/lidarclip/extract_pc_features.py` | 多帧 LiDAR 特征提取与拼接 |
| `encoders/lidarclip/lidarclip/model/sst.py` | SST 编码器 |
| `encoders/lidarclip/lidarclip/model/attention_pool.py` | Attention Pooling |
| `datageneration/generate_dataset.py` | QA 数据生成 |
| `datageneration/prompts.py` | prompt 模板 |
| `model.py` | LiDAR-LLM 原始模型（含 VAT） |
| `mllm/vtimellm/eval/eval.py` | 评估脚本 |
