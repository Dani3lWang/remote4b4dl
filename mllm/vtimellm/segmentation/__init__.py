"""Language-guided point segmentation for B4DL.

The package is deliberately separate from the maintained B3 text path.  A
segmentation model is assembled only through :func:`load_reasonseg_model`, so
loading an older B3 checkpoint continues to use ``vtimellm.model.builder``.
"""

from .config import ReasonSegConfig
from .heads import HierarchicalMaskDecoder, ReasonSegHeadOutput
from .tokens import (
    B4DL_SPECIAL_TOKENS,
    SEGMENTATION_SPECIAL_TOKENS,
    SegmentationTokenIds,
    register_segmentation_tokens,
)

__all__ = [
    "B4DL_SPECIAL_TOKENS",
    "SEGMENTATION_SPECIAL_TOKENS",
    "HierarchicalMaskDecoder",
    "ReasonSegConfig",
    "ReasonSegHeadOutput",
    "SegmentationTokenIds",
    "register_segmentation_tokens",
]
