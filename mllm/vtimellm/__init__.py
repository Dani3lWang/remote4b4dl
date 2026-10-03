"""Keep data viewers usable without importing the optional model stack."""

__all__ = ["VTimeLLMLlamaForCausalLM"]


def __getattr__(name):
    if name == "VTimeLLMLlamaForCausalLM":
        from .model import VTimeLLMLlamaForCausalLM
        return VTimeLLMLlamaForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
