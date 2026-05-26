# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.

import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
print(root_dir)
import sys
sys.path.append(root_dir)

# flash_attn 2.x 不支持 RTX 5090 (sm_120)，改用 transformers 原生 SDPA
# 需要 transformers >= 4.45
try:
    from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    replace_llama_attn_with_flash_attn()
except ImportError:
    pass  # flash_attn not installed, using transformers native SDPA

from train import train

if __name__ == "__main__":
    train()
