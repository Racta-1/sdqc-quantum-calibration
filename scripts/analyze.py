#!/usr/bin/env python
"""Rebuild all paper tables, SUMMARY.md and the hard-case figure from a results folder.

  python scripts/analyze.py --results results/v5_1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdqc.analysis import analyze  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/v5_1")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    analyze(a.results, a.n_boot)
