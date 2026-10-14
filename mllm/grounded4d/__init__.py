"""Independent, point-preserving query-mask training for B4DL.

This package deliberately does not import VTimeLLM, Vicuna or CUDA extensions
at import time. Language alignment uses separately recorded phrase embeddings.
"""
