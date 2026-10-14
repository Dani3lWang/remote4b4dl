#!/usr/bin/env python3
"""Evaluate predicted spatial instance masks on a complete development manifest."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounded4d.evaluate import main


if __name__ == "__main__":
    main()
