#!/usr/bin/env python3
"""Independent B3 answer segmentation interface; baseline scripts are unchanged."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from answerseg.runner import main

if __name__ == "__main__":
    main()
