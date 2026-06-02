"""Launch the pinned SAM 3 training entry point with a project-relative path."""

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "vendor"))
runpy.run_path(
    str(PROJECT_ROOT / "vendor" / "sam3" / "train" / "train.py"),
    run_name="__main__",
)
