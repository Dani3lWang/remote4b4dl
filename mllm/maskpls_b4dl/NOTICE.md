# 来源与移植边界

本目录依据 PRBonn/MaskPLS 的 MIT 许可进行移植：

- 仓库：https://github.com/PRBonn/MaskPLS
- 固定提交：`84881ba83f2f2e846cbded677544993d0c8ecd76`
- 论文：Rodrigo Marcuzzi et al., Mask-Based Panoptic LiDAR Segmentation for Autonomous Driving, IEEE RA-L 2023, DOI 10.1109/LRA.2023.3236568。
- 原始许可证保存在 `LICENSE-MaskPLS.txt`。
- 原始 blocks/decoder 来自 Mask2Former 的修改版，保留其来源注释及 `LICENSE-Mask2Former.txt`。

直接派生的文件：`backbone.py`（上游 mink.py）、`decoder.py`、`blocks.py`、`positional_encoder.py`。
插值、集合匹配、BCE/Dice、语义 CE/Lovasz 按上游方法重新组织到本项目。其余文件为 B4DL 的数据、训练、协议与对象接口适配。

保留完整的 4 级稀疏编码器/解码器、3 尺度 × 3 decoder blocks、256 维对象 query、辅助层监督、语义辅助头、逆平方距离 KNN 插值和位置编码。保留参数名以及官方 nuScenes 分类行：0=ignore，1..16=语义类别，17=no-object；辅助语义头保留上游的 20 行以兼容权重，目标标签只使用 1..16。

本移植的差异：

1. 去除 Lightning/Open3D/转换后 KITTI 文件依赖，直接读取 B4DL 已审核的原始 nuScenes manifest。
2. KNN 显式插值回输入原始点的行序；支持不足 3 个体素时的有效邻居数量及分块 Torch 诊断实现。
3. Attention 使用 `need_weights=False`，可进行 decoder activation checkpointing；参数形状不变，浮点输出不承诺逐位相同。
4. 单卡训练默认 128 queries；官方冻结推理保持 100。训练初始化扩展时，仅复制已有 query 行并保留新增行的随机初始化，严格核对其他权重。
5. 训练保留全部可见 Thing 实例（含 1 点目标）及 6 类 Stuff 的监督；采样至少覆盖每个实例的一个前景点；评测不使用上游的 50 点 FP/FN 排除。
6. 输出使用 softmax 类别分数；overlap 采用阈值化掩码中实际分配给该 query 的比例。因此输出处理不是上游 raw-logit 加权的逐位复现。
7. 主指标为 B4DL 的类别正确、一对一实例召回/精确率与全部 GT 平均 IoU，不冒充官方 PQ。候选 oracle 诊断单列。
8. 保留对象特征及稳定的帧内 ID；不把帧内 query ID 当作轨迹 ID，不将候选类别当作回答指代绑定。
9. 当前默认保持原始坐标，不做随机增强；正式泛化训练所需的增强须另行核对 ROI、点序与完整实例保留。

nuScenes 原始坐标与上游转换器的一致性依据：
https://github.com/PRBonn/nuscenes2kitti/blob/5bfe90069820a4276c62bcbc1aaf00314957c20a/nuscenes2kitti.py
该转换器直接保留 XYZ/intensity 的点序，没有旋转；`--normalize_remission` 是可选项。
本移植将 intensity_mode 写入配置及 checkpoint（raw/frame_minmax），常量强度归一化时返回 0 避免 NaN。
仅凭公开权重无法确定作者实际是否开启了该归一化，加载记录明确保留 preprocessing_equivalence_verified=false。

这是一份面向 B4DL 的适配版本，不宣称复现论文数值、实现 Mask4D 跟踪或完成 B3 自由生成到实例的训练。
