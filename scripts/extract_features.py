#!/usr/bin/env python
"""Download MedMNIST and cache frozen ResNet-18 features (the step that benefits most from a GPU).

  python scripts/extract_features.py --device cuda
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdqc.config import make_config  # noqa: E402
from sdqc.data import FeatureExtractor, build_dataset, resolve_device  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["pneumoniamnist", "breastmnist", "octmnist"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--batch-size", type=int, default=256)
    a = ap.parse_args()
    cfg = make_config(a.quick)
    dev = resolve_device(a.device)
    print("Feature device:", dev)
    from sdqc.data import extract_or_load_features
    ext = FeatureExtractor(dev, a.batch_size)
    for flag in a.datasets:
        extract_or_load_features(flag, cfg, a.cache, dev, extractor=ext)
        build_dataset(flag, cfg, a.cache)          # prints split sizes / prevalence
