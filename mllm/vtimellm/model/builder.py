import os
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from vtimellm.model import *
from peft import PeftModel

def load_lora(model, lora_path):
    non_lora_trainables_path = os.path.join(lora_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_trainables_path):
        non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, lora_path)
    return model

def load_pretrained_model(args, stage2=None, stage3=None):
    kwargs = {'torch_dtype': torch.float16}

    # model_path = os.path.expanduser(args.model_path)
    model_base = args.model_base


    # lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
    print('Loading VTimeLLM from base model...')
    if 'chatglm' in model_base:
        tokenizer = AutoTokenizer.from_pretrained(model_base, trust_remote_code=True)
        model = VTimeLLMChatGLMForCausalLM.from_pretrained(model_base)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        model = VTimeLLMLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

    # ── Register B4DL special tokens at inference time (paper §4.1) ──
    # <4DLiDAR> and <meta> must be single tokens for the model to use
    # dedicated embeddings. If the model was already trained with these
    # tokens (embeddings already resized), add_special_tokens is a no-op.
    _b4dl_tokens = ["<4DLiDAR>", "<meta>"]
    _num_new = tokenizer.add_special_tokens(
        {"additional_special_tokens": _b4dl_tokens})
    if _num_new > 0:
        # Tokens are new — resize embeddings and initialise with mean
        print(f"  Registered {_num_new} new B4DL special tokens, resizing embeddings...")
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            _emb = model.get_input_embeddings().weight.data
            _out_emb = model.get_output_embeddings().weight.data
            _avg_in = _emb[:-_num_new].mean(dim=0, keepdim=True)
            _avg_out = _out_emb[:-_num_new].mean(dim=0, keepdim=True)
            _emb[-_num_new:] = _avg_in
            _out_emb[-_num_new:] = _avg_out


    # load stage1:
    model.get_model().initialize_vision_modules(args)

    if stage2 is not None:
        print('Loading stage2 weights...')
        model = load_lora(model, stage2)
        print('Merging stage2 weights...')
        model = model.merge_and_unload()
        if stage3 is not None:
            print('Loading stage3 weights...')
            model = load_lora(model, stage3)
            print('Merging stage3 weights...')
            model = model.merge_and_unload()


    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
