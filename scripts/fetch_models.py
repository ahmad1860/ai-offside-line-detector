"""Explicitly download pretrained YOLO weights into the local ``models/`` dir.

Run this once (requires internet) before offline use:

    python scripts/fetch_models.py                # fetch default yolo11n.pt
    python scripts/fetch_models.py --model yolo11n-pose.pt

Weights are gitignored. Nothing is downloaded implicitly elsewhere.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Make the src/ layout importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offside.config import load_config


def fetch(model_name: str, models_dir: Path) -> Path:
    """Download ``model_name`` via Ultralytics and place it in ``models_dir``."""
    from ultralytics import YOLO

    models_dir.mkdir(parents=True, exist_ok=True)
    # Instantiating YOLO with a bare name triggers the download to CWD/cache.
    model = YOLO(model_name)
    # Locate the resolved weights file and copy it into models/.
    ckpt = Path(getattr(model, "ckpt_path", "") or model_name)
    target = models_dir / model_name
    if ckpt.is_file() and ckpt.resolve() != target.resolve():
        shutil.copy2(ckpt, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pretrained YOLO weights.")
    parser.add_argument("--model", default=None, help="Model file name, e.g. yolo11n.pt")
    args = parser.parse_args()

    cfg = load_config()
    model_name = args.model or cfg.detection.model_name
    target = fetch(model_name, cfg.paths.models)
    print(f"Model available at: {target}")


if __name__ == "__main__":
    main()
