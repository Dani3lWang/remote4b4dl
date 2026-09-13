"""Model assembly shared by the three RL processes.

Reuses `vtimellm.model.builder.load_pretrained_model` (base + stage1 projector
+ merged stage2 LoRA) instead of re-implementing it; the two additions are the
load dtype and the left-padding settings that batched `generate` needs.
"""

import torch
from easydict import EasyDict as edict

from vtimellm.constants import IMAGE_TOKEN_INDEX
from vtimellm.conversation import conv_templates, SeparatorStyle
from vtimellm.mm_utils import tokenizer_image_token
from vtimellm.model.builder import load_pretrained_model

from . import DEFAULT_MODEL_BASE, DEFAULT_MM_ADAPTER

# B3's adapter targets exactly these (checkpoints/...-b3/adapter_config.json);
# the RL adapter must match so the two LoRAs are comparable when merged.
RL_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                     'gate_proj', 'up_proj', 'down_proj']


def load_merged_actor(stage2, dtype=torch.bfloat16,
                      model_base=DEFAULT_MODEL_BASE,
                      mm_adapter=DEFAULT_MM_ADAPTER,
                      device='cuda'):
    """base + stage1 projector + stage2 LoRA merged, resident on one GPU.

    dtype is uniform on purpose: the rollout, reference and training processes
    must agree, otherwise `exp(logp_actor - logp_old)` carries a systematic
    precision offset instead of measuring policy movement.
    """
    args = edict(dict(model_base=model_base,
                      pretrain_mm_mlp_adapter=mm_adapter,
                      torch_dtype=dtype))
    tokenizer, model, context_len = load_pretrained_model(args, stage2=stage2)
    if getattr(model.config, 'use_frame_position_embedding', False):
        raise ValueError(
            f'{stage2} enables frame-position embeddings; the B3 baseline does '
            'not, and this stack passes frame_indices=None')
    prepare_for_generation(tokenizer, model)
    model = model.to(dtype)
    if device:
        model = model.to(device)
    model.eval()
    return tokenizer, model, context_len


def prepare_for_generation(tokenizer, model):
    """Batched generate needs left padding and a cache_position fix.

    `builder` never touches padding_side and `train.py:343` sets "right" for the
    SFT collator. Right padding is wrong here: the fused first forward returns
    logits for the re-padded sequence, and `generate` reads position -1, which
    for a right-padded row is a pad rather than the last real token.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token   # same choice as train.py:346
    tokenizer.padding_side = 'left'
    # arch.py:244 re-pads the fused sequence using this config key, default
    # 'right'. Left padding is also what makes the per-step mask rebuild in
    # arch.py:89-94 land on the right positions (see data.iter_prompt_batches).
    model.config.tokenizer_padding_side = 'left'
    _drop_cache_position(model)


def _drop_cache_position(model):
    """Make batched `generate` survive the multimodal expansion.

    transformers derives `cache_position` from the *unexpanded* input_ids, but
    arch.py replaces the single `<video>` placeholder with N embeddings, so
    `_update_causal_mask` is handed target_length = L + N - 1 against a
    cache_position of length L and raises a broadcast error. Batch=1 escapes it
    only because an unpadded, all-ones mask takes the sdpa fast path -- which is
    why `inference.py` has never hit this.

    Dropping the value lets LlamaModel.forward recompute it from the real
    `inputs_embeds` length: arange(0, E) on the first step and past_seen + 1 on
    every later step, both correct.

    Patched on the instance, not in vtimellm_llama.py, so the frozen evaluation
    path (test_b4dl -> load_pretrained_model) is provably untouched.
    """
    original = model.prepare_inputs_for_generation

    def patched(input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        model_inputs = original(input_ids, past_key_values=past_key_values,
                                inputs_embeds=inputs_embeds, **kwargs)
        model_inputs.pop('cache_position', None)
        return model_inputs

    model.prepare_inputs_for_generation = patched


def attach_rl_lora(model, r=64, alpha=128, dropout=0.0):
    """Freeze the merged SFT weights and train a fresh LoRA on top.

    Mirrors the stage-3 path in train.py:384-392 except that the SFT adapter is
    already merged into `model`. dropout must be 0: a stochastic actor makes
    logp_actor irreproducible between the forward and any recomputation, which
    biases the importance ratio.

    The <4DLiDAR>/<meta> embedding rows stay frozen (B3's trained rows are
    already inside the merged weights), so the gradient-mask hook and the
    weight_decay=0 constraint from train.py:399-409 do not apply here.
    """
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias='none',
                     task_type='CAUSAL_LM', target_modules=RL_TARGET_MODULES)
    return get_peft_model(model, cfg)


def stop_string():
    """Generation terminator for the v1 template (matches inference.py:41)."""
    conv = conv_templates['v1'].copy()
    return conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2


def encode_prompt(tokenizer, human_value):
    """Prompt ids for one training entry, byte-identical to the SFT loader.

    dataset.py:313 feeds `conversations[i].value` straight into the v1 template
    and inference.py:38-40 does the same for a query, so passing the stored
    human value through unchanged is what keeps RL prompts equal to SFT prompts.
    """
    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], human_value)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                return_tensors='pt')
    return prompt, ids, stop_string()
