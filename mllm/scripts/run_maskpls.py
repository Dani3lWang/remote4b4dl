#!/usr/bin/env python3
"""Native MaskPLS spatial backend for B4DL (no B3/Vicuna import)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maskpls_b4dl.runner import main

if __name__ == "__main__":
    main()
