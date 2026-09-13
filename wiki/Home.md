# B4DL

本仓库维护 ACM Multimedia 2025 B4DL 的官方基础模块，以及当前唯一 SFT 基线 **B3**。

- B3：整场景 LiDAR 输入 + relative-to-previous meta2。
- 当前训练配方：2 epochs。
- 历史 3-epoch B3 checkpoint 的 mIoU 0.3467 仅作历史对照。
- 强化学习代码位于 `rf-grpo` 分支。

快速入口：

- [[Installation]]
- [[Architecture]]
- [[Data-Generation]]
- [[LiDAR-CLIP-Encoder]]
- [[Training]]
- [[Inference-and-Evaluation]]
- [[FAQ-and-Known-Issues]]
- [[Repository-Structure]]
