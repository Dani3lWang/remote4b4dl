# B4DL 项目待办事项

> 生成日期: 2026-05-30
> 项目当前状态: 三阶段训练完成，Stage2/Stage3 评估完成

---

## 一、模型性能提升（高优先级）

### 1. 继续 Stage3 训练（未完全收敛）
- 当前 Stage3 Loss 从 2.82 → 1.35（仅下降 52%），仍有很大下降空间
- 增加 epochs: 3 → 5-10
- 提升学习率: 2e-5 → 5e-5
- 调整 LoRA rank: 64 → 128

### 2. 改善时序定位（Frame Range IoU 仅 17.11%，最大短板）
- 增加时序位置编码（当前帧级细腻度被压缩为 N×768 矩阵）
- 引入帧级 attention mask 增强时间感知
- 调整训练数据中 Frame Range 样本权重

### 3. 提升类别理解（Categorical 准确率仅 39.22%）
- 数据增强平衡类别分布
- 对罕见类别（如 Wheelchair）增加训练样本
- 尝试 class-balanced loss weighting

### 4. 超参数消融实验

| 实验 | 说明 |
|------|------|
| LoRA rank 对比 | r=32 vs 64 vs 128 |
| 学习率搜索 | 不同阶段最优 lr |
| LoRA merge 策略验证 | 对比不 merge 直接 train Stage3（验证 catastrophic forgetting） |
| Epoch 数量 | Stage2 和 Stage3 最佳 epoch 数 |

---

## 二、评估体系扩展

### 5. 引入 LLM-based 评估（GPT-score）
- 项目中 `requirements_sum/config.py` 有评估配置
- 接入 GPT-score 作为 n-gram 指标的补充
- 解决开放式描述不存在唯一正确答案的天花板问题

### 6. 补充 CIDEr / SPICE 指标
- `pycocoevalcap` 已安装，当前只用了 BLEU/ROUGE-L/METEOR
- 补上 CIDEr 和 SPICE 提供更全面的 captioning 评估

### 7. 跨阶段对比分析
- 对比 Stage2-only vs Stage2+Stage3 在 QA 任务上的退化（catastrophic forgetting check）
- 分析哪些场景类型模型表现最好/最差
- 按场景复杂度（车辆数、行人数）分层分析性能

### 8. 简单基线对比
- Random baseline / majority-class baseline 对比
- "always say yes" 在 binary QA 上的表现
- Mean-pooled features + MLP classifier 基线

---

## 三、Demo 与可交互应用

### 9. 启动 Gradio Web Demo
项目已有 `mllm/vtimellm/demo_gradio.py`:

```bash
cd /root/autodl-tmp/wql/mmb4dl/mllm
conda run -n b4dl python vtimellm/demo_gradio.py
```

### 10. 构建批量推理 Pipeline
- 创建脚本批量处理任意 nuScenes 场景
- 可视化输出（点云 + 文本描述并排展示）
- 支持自定义 query 交互式推理

---

## 四、多模型对比实验

### 11. ChatGLM 替代实验
- 已有 `stage1_glm.sh` / `stage2_glm.sh`
- 对比 ChatGLM vs Vicuna 在 LiDAR 理解上的差异
- 分析不同 LLM 架构对 4D LiDAR 理解的影响

### 12. 更大模型迁移
- Vicuna-13B 或 LLaMA-3-8B 替换 7B 基座
- 观察模型容量对时序定位和类别理解的提升
- 评估显存/性能的 trade-off

---

## 五、数据管线

### 13. 生成更多训练数据
- `datageneration/` 管道支持 5 种任务类型: existence / binary / description / temporal / comprehensive
- 针对不足的任务类型（temporal、categorical）定向生成更多数据
- 平衡类别分布

### 14. 扩展到其他自动驾驶数据集
- 将相同 pipeline 应用到 Waymo Open Dataset
- 或 KITTI 数据集
- 验证方法的跨数据集泛化能力

---

## 六、论文与写作

### 15. 撰写实验分析论文
- 基于 `PAPER_PLAN.md` 大纲撰写 LaTeX 正文
- 使用 `/paper-write` 和 `/paper-compile` 命令

### 16. 整理补充材料
- 绘制更清晰的 Loss 曲线图
- Stage2 vs Stage3 的并排可视化
- Failure case 分析（定性错误案例展示）

---

## 七、工程与代码质量

### 17. 清理硬编码 API key
- `datageneration/config.py` 含硬编码 OpenAI API key
- CLAUDE.md 明确标注需清理，勿提交到公开仓库

### 18. 添加单元测试
- 当前项目无测试文件
- 评估脚本（`b4dl_eval.py` / `b4dl_metrics.py`）的正确性依赖手动验证
- 建议：数据加载测试、指标计算正确性测试、推理输出格式验证

### 19. 依赖版本整理
- `requirements.txt` vs `requirements_b4dl.txt` 存在版本冲突说明
- 统一整理为清晰的依赖文件
- 标注 Python/CUDA 版本兼容性

### 20. 代码文档与注释
- 关键函数补充 docstring
- 训练和评估流程的 README 更新
- 添加常见问题 (FAQ) 文档

---

## 优先级建议

| 优先级 | 事项 | 理由 |
|--------|------|------|
| 🔴 P0 | #2 时序定位优化 | 最大短板，提升空间最大 |
| 🔴 P0 | #8 简单基线对比 | 论文必需，当前完全缺失 |
| 🟡 P1 | #1 继续 Stage3 训练 | 直接提升描述生成质量 |
| 🟡 P1 | #4 LoRA merge 消融实验 | 验证核心设计决策 |
| 🟢 P2 | #5 LLM-based 评估 | 补充 n-gram 指标不足 |
| 🟢 P2 | #9 Gradio Demo | 展示项目可用性 |
| 🔵 P3 | #11-12 多模型对比 | 扩展性实验 |
| 🔵 P3 | #17-20 工程质量 | 长期维护 |
