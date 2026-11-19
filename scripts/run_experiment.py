#!/usr/bin/env python
"""Run the SDQC benchmark (V5.1 design).

Examples
--------
  # smoke test (~5 min on CPU)
  python scripts/run_experiment.py --quick

  # full paper run, ResNet features on GPU, noisy circuits on GPU, 3 parallel workers
  python scripts/run_experiment.py --feature-device cuda --sim-device cuda --jobs 3

  # more seeds for the paper
  python scripts/run_experiment.py --seeds 0 1 2 3 4 5 6 7 8 9 --jobs 4

Each (dataset, seed) unit is written to <out>/raw/ as soon as it finishes, so an interrupted run
resumes where it stopped (use --overwrite to recompute).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from sdqc.config import make_config  # noqa: E402


def _unit(args):
    flag, seed, cfg, cache_dir, sim_device, raw_dir, threads = args
    import torch
    torch.set_num_threads(threads)
    from sdqc.circuits import Simulator
    from sdqc.data import build_dataset
    from sdqc.experiment import run_dataset_seed
    from sdqc.hardware import build_conditions
    D = build_dataset(flag, cfg, cache_dir, device="cpu", verbose=False)   # features already cached
    rows, sanity, probes = run_dataset_seed(flag, seed, D, Simulator(sim_device), cfg, build_conditions(cfg))
    tag = f"{flag}_seed{seed}"
    pd.DataFrame(rows).to_csv(raw_dir / f"{tag}_metrics.csv", index=False)
    pd.DataFrame(sanity).to_csv(raw_dir / f"{tag}_sanity.csv", index=False)
    pd.DataFrame(probes).to_csv(raw_dir / f"{tag}_probes.csv", index=False)
    return tag


def _legacy_run(cfg, a, raw, units):
    """Notebook-identical order: one process, one shared condition RNG across all units."""
    from sdqc.circuits import Simulator
    from sdqc.data import build_dataset
    from sdqc.experiment import run_dataset_seed
    from sdqc.hardware import build_conditions
    from sdqc.hardware import reported
    conds, sim = build_conditions(cfg), Simulator(a.sim_device)
    for _, c in conds["TEST_CONDS"]:      # the notebook drew these once at setup (TEST_REPORTED), advancing the stream
        reported(c, conds["rc"], cfg["REPORT_ERROR"])
    for flag in cfg["DATASETS"]:
        D = build_dataset(flag, cfg, a.cache, device="cpu", verbose=False)
        for seed in cfg["SEEDS"]:
            rows, sanity, probes = run_dataset_seed(flag, seed, D, sim, cfg, conds, legacy_report_stream=True)
            tag = f"{flag}_seed{seed}"
            pd.DataFrame(rows).to_csv(raw / f"{tag}_metrics.csv", index=False)
            pd.DataFrame(sanity).to_csv(raw / f"{tag}_sanity.csv", index=False)
            pd.DataFrame(probes).to_csv(raw / f"{tag}_probes.csv", index=False)
            print(f"  saved {tag}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small smoke-test configuration")
    ap.add_argument("--datasets", nargs="+", choices=["pneumoniamnist", "breastmnist", "octmnist"])
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--probe-n", type=int, help="number of SDQC probes (default 96)")
    ap.add_argument("--out", default="results/v5_1", help="output folder")
    ap.add_argument("--cache", default="cache", help="feature cache folder")
    ap.add_argument("--feature-device", default="auto", help="auto | cpu | cuda | cuda:0 ... (ResNet-18 features)")
    ap.add_argument("--sim-device", default="cpu", help="cpu (NumPy, as in paper) | cuda | cuda:0 ... (noisy circuits)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel (dataset, seed) workers")
    ap.add_argument("--overwrite", action="store_true", help="recompute units that already exist")
    ap.add_argument("--legacy-report-stream", action="store_true",
                    help="reproduce the notebook exactly (sequential reported-rate stream; forces --jobs 1 and recomputes all units)")
    ap.add_argument("--no-analysis", action="store_true", help="skip building the tables at the end")
    a = ap.parse_args()

    cfg = make_config(a.quick, DATASETS=a.datasets, SEEDS=a.seeds, PROBE_N=a.probe_n)
    out = Path(a.out); raw = out / "raw"; raw.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    import torch
    from sdqc.data import FeatureExtractor, cache_path, extract_or_load_features, resolve_device
    print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    fdev = resolve_device(a.feature_device)
    print(f"ResNet features on: {fdev} | noisy circuits on: {a.sim_device} | workers: {a.jobs}")

    # 1) features (GPU-heavy part) in the main process, cached to disk
    extractor = None
    for flag in cfg["DATASETS"]:
        if not os.path.exists(cache_path(a.cache, flag, cfg)):
            extractor = extractor or FeatureExtractor(fdev)
            extract_or_load_features(flag, cfg, a.cache, fdev, extractor=extractor)
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2) (dataset, seed) units, resumable
    units = [(f, s) for f in cfg["DATASETS"] for s in cfg["SEEDS"]]
    if a.legacy_report_stream:
        a.jobs, a.overwrite = 1, True
    todo = [u for u in units if a.overwrite or not (raw / f"{u[0]}_seed{u[1]}_metrics.csv").exists()]
    print(f"{len(units)} units, {len(units) - len(todo)} already done, running {len(todo)}")
    threads = max(1, (os.cpu_count() or 1) // max(1, a.jobs))
    args = [(f, s, cfg, a.cache, a.sim_device, raw, threads) for f, s in todo]
    if a.legacy_report_stream:
        _legacy_run(cfg, a, raw, units)
    elif a.jobs > 1 and len(args) > 1:
        with mp.get_context("spawn").Pool(a.jobs) as pool:
            for tag in pool.imap_unordered(_unit, args):
                print(f"  saved {tag}", flush=True)
    else:
        for x in args:
            print(f"  saved {_unit(x)}", flush=True)

    # 3) merge
    def merge(suffix):
        files = [raw / f"{f}_seed{s}_{suffix}.csv" for f, s in units]
        return pd.concat([pd.read_csv(p) for p in files if p.exists()], ignore_index=True)
    merge("metrics").to_csv(out / "metrics.csv", index=False)
    merge("sanity").to_csv(out / "sanity.csv", index=False)
    merge("probes").to_csv(out / "probe_diagnostics.csv", index=False)
    with open(out / "config.json", "w") as f:
        json.dump(dict(CFG=cfg, feature_device=fdev, sim_device=a.sim_device), f, indent=2, default=str)
    print(f"Merged results in {out} ({(time.time() - t_all) / 60:.1f} min)")

    if not a.no_analysis:
        from sdqc.analysis import analyze
        analyze(out, n_boot=cfg["N_BOOT"])


if __name__ == "__main__":
    main()
