# Models directory

Model weights are **not committed** to Git (see `.gitignore`). Download them once, on your
initiative, with:

```powershell
python scripts/fetch_models.py
```

## Baseline

- `yolo11n.pt` (about 5.4 MB) — the smallest YOLO11 detector, chosen for CPU-first inference.

## Notes

- Weights are saved to this directory. Ultralytics may also leave a copy in the folder you ran the
  command from; Git ignores it and it can be deleted.
- No weights are downloaded implicitly at import time or by the tests.
- YOLO11 weights are distributed by Ultralytics under AGPL-3.0 (or a commercial license). Check the
  terms before redistributing them.
- Pose weights (`yolo11n-pose.pt`) are optional and not used by the current version.
