"""Version/API preflight without downloading or loading pretrained weights."""
import argparse
from importlib import metadata
import inspect
import json
import sys


VERSIONS = {"torch": "2.8.0", "transformers": "4.47.0", "peft": "0.13.2",
            "accelerate": "1.15.0", "numpy": "1.26.4", "sentencepiece": "0.2.2"}


def runtime_identity(model):
    """Bind wheel variants and generation defaults as well as model weights."""
    import torch
    return {"python": sys.version.split()[0],
            "packages": {name: metadata.version(name) for name in VERSIONS},
            "torch_cuda": torch.version.cuda,
            "attention_implementation": model.config._attn_implementation,
            "generation_config": model.generation_config.to_dict()}


def check_runtime(require_cuda=False):
    problems, installed = [], {}
    if sys.version_info[:2] != (3, 10):
        problems.append("Python 3.10 required for the validated upcore stack")
    for name, expected in VERSIONS.items():
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed[name] = None
        if installed[name] is None or installed[name].split("+")[0] != expected:
            problems.append(f"{name}=={expected} required, installed={installed[name]}")
    if problems:
        raise RuntimeError("upcore runtime preflight failed: " + "; ".join(problems)
                           + ". See mllm/docs/upcore_runtime.md")
    try:
        import torch
        import peft
        import accelerate
        import sentencepiece
        from transformers import LlamaForCausalLM
        if "cache_position" not in inspect.signature(LlamaForCausalLM.forward).parameters:
            raise RuntimeError("LlamaForCausalLM.forward lacks cache_position")
        if not callable(torch.nn.functional.scaled_dot_product_attention):
            raise RuntimeError("PyTorch SDPA unavailable")
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    except Exception as error:
        raise RuntimeError("upcore runtime API preflight failed: " + str(error)) from error
    return {"python": sys.version.split()[0], "packages": installed,
            "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check_runtime(args.require_cuda), indent=2))


if __name__ == "__main__":
    main()
