#!/usr/bin/env python
"""V4 Part B figure: Brier vs held-out depolarizing level, per model (paper Fig. 3).

  python scripts/plot_v4_partB.py --csv paper_results/v4/v4_partB.csv --dataset pneumoniamnist
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["pdf.fonttype"] = 42   # embed TrueType (IEEE PDF eXpress rejects Type 3 fonts)
plt.rcParams["font.family"] = "DejaVu Sans"
METHODS = ["Fixed Platt", "Noise-conditioned Platt", "Physics-informed", "Moment matching", "Oracle Platt"]

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="paper_results/v4/v4_partB.csv")
ap.add_argument("--dataset", default="pneumoniamnist")
ap.add_argument("--out", default="docs/figures/v4_partB_{dataset}")
a = ap.parse_args()

r = pd.read_csv(a.csv)
d = r[(r.dataset == a.dataset) & (r.eval_set == "test") & (r.family == "held-out depolarizing")]
names = list(d.model.unique())
fig, axes = plt.subplots(1, len(names), figsize=(4.3 * len(names), 3.8), sharey=True, squeeze=False)
for ax, name in zip(axes[0], names):
    for meth in METHODS:
        s_ = d[(d.model == name) & (d.method == meth)].groupby("p")["Brier"].mean()
        ax.plot(s_.index, s_.values, "k-" if meth == "Oracle Platt" else "o-", label=meth)
    ax.set(xscale="log", title=name, xlabel="depolarizing p")
axes[0][0].set_ylabel("Brier (lower is better)"); axes[0][0].legend(fontsize=7)
fig.suptitle(a.dataset); plt.tight_layout()
out = a.out.format(dataset=a.dataset)
for ext in ("pdf", "png"):
    plt.savefig(f"{out}.{ext}", dpi=150)
print("saved", out + ".pdf")
