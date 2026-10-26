"""Read-only, layer-at-a-time CPU replay of the four A3 failure candidates.

No optimizer, backward pass, training entry point or checkpoint write is used.
This is a CPU diagnostic of step 4750, not a replay of the lost step-4953 state.
"""
import argparse
import ctypes
import gc
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding, apply_rotary_pos_emb

from b4dl_core.b3 import batch_indices
from b4dl_core.inputs import GeometryDataset
from b4dl_core.integration import attach, batch_inputs, restore_trainable
from b4dl_core.io import read_json, sha256
from vtimellm.model.vtimellm_llama import VTimeLLMConfig, VTimeLLMLlamaForCausalLM


def emit(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def stats(value):
    finite = torch.isfinite(value)
    return {"dtype": str(value.dtype), "shape": list(value.shape), "finite": bool(finite.all()),
            "nonfinite": int((~finite).sum()),
            "finite_abs_max": float(value[finite].abs().max()) if finite.any() else None}


def decoder_forward(config, hidden, rotary, linear, normalize):
    """The SDPA LlamaDecoderLayer path, with one resident linear weight."""
    position_ids = torch.arange(hidden.shape[1]).unsqueeze(0)
    normalized = normalize("input_layernorm", hidden)
    head_dim = config.hidden_size // config.num_attention_heads
    shape = (*hidden.shape[:2], config.num_attention_heads, head_dim)
    q, k, v = [linear("self_attn."+name+"_proj", normalized).view(shape).transpose(1, 2)
               for name in ("q", "k", "v")]
    cos, sin = rotary(hidden, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    attended = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    attended = attended.transpose(1, 2).contiguous().reshape_as(hidden)
    hidden = hidden + linear("self_attn.o_proj", attended)
    normalized = normalize("post_attention_layernorm", hidden)
    gate = torch.nn.functional.silu(linear("mlp.gate_proj", normalized))
    up = linear("mlp.up_proj", normalized)
    return hidden + linear("mlp.down_proj", gate * up)


@torch.no_grad()
def replay(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    lock = read_json(args.baseline)
    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("checkpoint hash mismatch")
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if sha256(args.baseline) != bundle["identity"]["baseline_manifest_sha256"]:
        raise ValueError("baseline manifest mismatch")
    dataset = GeometryDataset(args.dataset, "train")
    if dataset.manifest["dataset_id"] != bundle["identity"]["dataset_id"]:
        raise ValueError("dataset identity mismatch")
    emit(event="checkpoint", sha256=args.checkpoint_sha256, training=bundle["training"],
         identity=bundle["identity"], cuda_available=torch.cuda.is_available(), optimizer_updates=0)
    base = Path(lock["components"]["model_base"]["path"])
    stage2 = Path(lock["components"]["stage2"]["path"])
    weight_map = read_json(base/"pytorch_model.bin.index.json")["weight_map"]
    shards = {}

    def weight(name):
        shard = weight_map[name]
        if shard not in shards:
            shards[shard] = torch.load(base/shard, map_location="cpu", weights_only=True, mmap=True)
        return shards[shard][name]

    # Match builder.load_pretrained_model: CPU FP16 base, resized output rows
    # initialised to the old-row mean, B3 input embeddings, then merged LoRA.
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=False, local_files_only=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<4DLiDAR>", "<meta>"]})
    config = VTimeLLMConfig.from_pretrained(base, local_files_only=True)
    config.vocab_size = len(tokenizer)
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = VTimeLLMLlamaForCausalLM(config)
        model.get_model().mm_projector = torch.nn.Linear(768, config.hidden_size)
    embeddings = torch.load(stage2/"non_lora_trainables.bin", map_location="cpu", weights_only=True, mmap=True)
    model.get_input_embeddings().weight = torch.nn.Parameter(
        embeddings["base_model.model.model.embed_tokens.weight"].to(torch.float16), requires_grad=False)
    model.get_model().mm_projector.to_empty(device="cpu")
    model, options = attach(model, arm="A3")
    if options != bundle["options"]:
        raise ValueError("this diagnostic expects the original A3 architecture")
    restore_trainable(model, bundle["parameters"])
    model.eval()
    cases = []
    for micro in range(4):
        index = batch_indices(len(dataset.requests), 4953, micro, 1, 4, 42)[0]
        request = dataset.requests[index]
        batch = batch_inputs(model, tokenizer, [dataset.model_input(request)], [dataset.targets[request["request_id"]]])
        _, positions, _, _, hidden, labels = model.prepare_inputs_labels_for_multimodal(
            batch["input_ids"], None, batch["attention_mask"], None, batch["labels"], batch["images"],
            **{k: batch[k] for k in ("frame_times", "frame_geometry", "frame_valid_mask")})
        cases.append({"micro": micro, "request_id": request["request_id"], "hidden": hidden, "labels": labels})
        emit(event="visual_expansion", micro=micro, request_id=request["request_id"],
             stats=stats(hidden), supervised_tokens=int(labels.ne(-100).sum()))
    del model, embeddings, bundle, dataset
    gc.collect()
    rotary = LlamaRotaryEmbedding(config=config)
    lora_config = read_json(stage2/"adapter_config.json")
    if lora_config["use_dora"] or lora_config["use_rslora"] or lora_config["rank_pattern"] or lora_config["alpha_pattern"]:
        raise ValueError("unsupported LoRA variant")
    scaling = lora_config["lora_alpha"] / lora_config["r"]
    with safe_open(str(stage2/"adapter_model.safetensors"), framework="pt", device="cpu") as lora:
        for index in range(config.num_hidden_layers):
            events = []

            def linear(name, hidden):
                key = f"model.layers.{index}.{name}.weight"
                value = weight(key).clone()
                stem = "base_model.model." + key.removesuffix(".weight")
                if stem+".lora_A.weight" in lora.keys():
                    # PeftModel.from_pretrained promotes B3 LoRA to FP32;
                    # merge adds its FP32 delta in place to the FP16 base.
                    a, b = lora.get_tensor(stem+".lora_A.weight"), lora.get_tensor(stem+".lora_B.weight")
                    for start in range(0, len(value), 128):
                        value[start:start+128].add_((b[start:start+128].float() @ a.float()) * scaling)
                output = torch.nn.functional.linear(hidden, value)
                events.append({"module": name, "input": stats(hidden), "output": stats(output)})
                del value
                ctypes.CDLL("libc.so.6").malloc_trim(0)
                return output

            def normalize(name, hidden):
                original_dtype = hidden.dtype
                values = hidden.float()
                values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + config.rms_norm_eps)
                return weight(f"model.layers.{index}.{name}.weight") * values.to(original_dtype)

            for case in cases:
                hidden = case["hidden"]
                events.clear()
                case["hidden"] = decoder_forward(config, hidden, rotary, linear, normalize)
                emit(event="decoder", layer=index, micro=case["micro"], stats=stats(case["hidden"]),
                     linears=list(events))
                del hidden
            gc.collect()
            # Return freed layer/delta allocations under small CPU cgroups.
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    norm.weight = torch.nn.Parameter(weight("model.norm.weight"), requires_grad=False)
    head = weight("lm_head.weight")
    for case in cases:
        hidden = norm(case["hidden"])
        logits = torch.nn.functional.linear(hidden, head)
        if config.vocab_size != len(head):
            extra = torch.nn.functional.linear(hidden, head.mean(0, keepdim=True)).expand(*logits.shape[:-1], config.vocab_size-len(head))
            logits = torch.cat((logits, extra), -1)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, config.vocab_size),
                                                 case["labels"][:, 1:].reshape(-1))
        emit(event="loss", micro=case["micro"], logits=stats(logits), loss=stats(loss),
             loss_value=float(loss) if torch.isfinite(loss) else None, optimizer_updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "dataset", "checkpoint", "checkpoint-sha256"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--threads", type=int, default=2)
    replay(parser.parse_args())
