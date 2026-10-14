#!/usr/bin/env python3
"""Run the first-layer attention control and its gated full-mini extension."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grounded4d.attention_study import main

if __name__ == "__main__":
    main()
