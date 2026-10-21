# 自由回答分割后插入 SEG：实现与真实输出检查

用户要求的顺序为：**B3 自由回答 → 推理分割 → 成功后插入 `<SEG>` → 文本与分割结果一起输出**。实现提交为 `2fc04b83eaac45e9d0a8dfd87dbc2cbd125408ce`，不要求 B3 自己生成 SEG。

## 已落实的接口

- `AnswerSegModel.predict()` 自动完成生成、分割、渲染；原问题使用普通 B3 提示。
- `segment_answer()` / CLI `segment-answer` 处理已有自由回答。默认将完整回答作为单个指代；多个目标接受纯文本字符 spans，最多八个，不接受 GT 实例 ID、query 或 mask。
- 内部 SEG probe 用于提取对齐状态，不作为输出标记。候选选择通过默认 0.5 门槛且掩码非空后，才在输出文本插入 SEG。
- 同时返回 `plain_answer/answer/objects/masks`。对象携带原文字符位置、输出标记位置及 `mask_index`；失败对象不占掩码行。CLI JSON 的 `mask_file/mask_index` 对应 NPZ 的原始点序掩码。
- 保留原文与标点。失败、低置信度、空掩码不插标记；原生 token 生成诊断另保留为 `predict_native()`。输出的 SEG 来源明确为 `post_segmentation`。

## 真实服务器检查

服务器为 `connect.bjb1.seetacloud.com:21358`，独立目录：

```text
/root/autodl-tmp/mmb4dl-answerseg-postseg-20261021-v1
```

源码归档 SHA256 为 `fdad5c0a4293b78c4649a1bec1eae4c0e3e7466afdcc8c61f0ab70ede3d0b982`，691 个归档文件逐一与部署目录核对通过。未覆盖原仓库或原训练。原 B3 训练 PID 72247 在检查后仍运行。

真实权重使用历史三轮 B3、Vicuna 7B、原 stage1 projector 和 P0 epoch-06 MaskPLS，身份记录在工程 bundle。语言模型 CPU 卸载、2 线程、nice 10，GPU 进程预算为总显存的 12.5%，进程限制 15 分钟。本次没有训练更新，**对齐头为随机初始化，明确标记未训练**；没有将它作为正式回答绑定模型。

输入为原数据中的问题、明确指定帧的 SST 特征和原始点云；不将原答案、候选实例 ID、真实掩码或固定 query 送入预测。单帧 `7fc4d2fe99ec49cabd6ee5822b717b6f` 有 34592 点，属于 trainval 工程样本，不是 mini 全量或独立留出评测。历史 SST 缓存的编码器 checkpoint 未独立核验。

| 检查 | 实际结果 |
|---|---|
| 本地 / 远端 AnswerSeg | 各 39 项通过 |
| 本轮空间 / 原回答绑定回归 | 65 / 8 项通过 |
| 原 B3 preflight / 真实输出进程 | 均退出 0 |
| 自由回答 | 正常生成，无原生 SEG；描述道路、汽车和行人 |
| 空间模型与选择器 | 真实执行；最高绑定置信度 0.3423188 |
| 默认门槛 | 0.5，未降低 |
| 输出状态 | `no_segment_selected / low_binding_confidence` |
| 输出文本 / 掩码 | 保留完整自由文本、无 SEG；NPZ `[0,34592]` |
| CUDA allocated / reserved 峰值 | 1287.15 / 1588 MiB |

该结果验证真实自由回答、真实空间前向、选择器拒绝、文本与空掩码一起输出的链路。**本次没有真实回答到正确实例的成功证据**。成功插入 SEG、非空掩码、多对象部分失败时的行号对应、中文标点和字符位置由显式 CPU 空间 fixture 检查覆盖；不能将 fixture 当作真实分割效果。

## 验收边界与证据

正式已审核 train/dev 回答绑定仍为零，待审核记录未升级。几何模板路线已依用户澄清从最终接口移除；先前失败目录保留，无可采纳的新模型。

紧凑证据归档 SHA256 为 `de25dcb0b3fd065ae61c56e02d83b5a637b702551a0c9b7944e31d789ef83715`，迁回 `training_logs/answerseg_20261021/deployment/postseg-remote/`。归档含 17 项清单证据和单帧原始点云；本地审计核对文件哈希、源码归档身份、源点云、NPZ 点序、文本保留与拒绝对象无输出标记。工程权重留远端，标明 `supervised_updates=0`、`binding_head_trained=false`、`model_performance_accepted=false`。

可复查：

- `mllm/answerseg/tests/test_postseg_output.py`
- `training_logs/answerseg_20261021/postseg-tests.log`
- `training_logs/answerseg_20261021/deployment/postseg-remote/postseg-local-audit.json`
- `training_logs/answerseg_20261021/deployment/postseg-remote/delivery/smoke-001/completion.json`
