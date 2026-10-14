#!/usr/bin/env python3
"""Filter manifests against original nuScenes panoptic classes before training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounded4d.filter import main


if __name__ == "__main__":
    main()
