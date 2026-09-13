# 安装

推荐使用 Python 3.10 的 `wqlc` Conda 环境。权威依赖文件为 `mllm/requirements.txt`。

```bash
conda create -n wqlc python=3.10 -y
conda activate wqlc
pip install -r mllm/requirements.txt
```

当前实测栈为 PyTorch 2.8.0+cu128、transformers 4.47.0、DeepSpeed 0.16.4、PEFT 0.13.2 和 accelerate 1.3.0。RTX 5090 环境使用 transformers 原生 SDPA，不依赖 flash-attn。

完整 B3 训练需要本地 Vicuna-7B、Stage1 projector、148K meta2 训练数据和按场景提取的 LiDAR 特征；这些大文件不应提交到 Git。
