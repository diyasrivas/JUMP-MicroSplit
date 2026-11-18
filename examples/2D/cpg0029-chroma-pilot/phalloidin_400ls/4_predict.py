#!/usr/bin/env python3
"""cpg0029 CP + Phenovue phalloidin 400LS (BR00122247). Step 4: prediction of the test FOVs + CellProfiler load_data.

Channels and paths: ../cpg0029_shared/dye_sets.py. Logic: ../cpg0029_shared/steps.py.
Run with --help for options.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpg0029_shared.steps import predict_main  # noqa: E402

if __name__ == "__main__":
    predict_main("phalloidin_400ls")
