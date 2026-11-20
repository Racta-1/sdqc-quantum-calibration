"""Turns raw metric rows into every table/figure used in the paper (CSV + a Markdown summary)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts (IEEE PDF eXpress rejects Type 3)
plt.rcParams["font.family"] = "DejaVu Sans"
import numpy as np
import pandas as pd

from .experiment import EST

SHOW = ["Fixed Platt", "Moment matching", "EM only", "Physics-informed + EM", "SDQC (probe only)",
        "SDQC-sim (probe + EM)", "SDQC (probe + EM)", "Oracle (noise + true prior)"]
COMPARATORS = ["Fixed Platt", "Moment matching", "EM only", "Physics-informed + EM", "SDQC (probe only)",
               "SDQC-sim (probe + EM)", "Oracle (noise + true prior)"]
# order in which the bootstrap RNG is consumed (kept identical to the notebook so CIs reproduce exactly)
_BOOT_ORDER = ["Fixed Platt", "Moment matching", "EM only", "Physics-informed + EM", "SDQC-sim (probe + EM)",
               "SDQC (probe only)", "Oracle (noise + true prior)"]
HARD = "unseen: amplitude damping"


def _prior_type(results):
    results = results.copy()
    results["prior_type"] = np.where(results.target_prior.astype(str) == "source",
                                     "same disease rate", "changed disease rate")
    return results


def brier_table(results):
    return (results[results.method.isin(SHOW)]
            .groupby(["dataset", "hardware_family", "prior_type", "method"])["Brier"].mean()
            .unstack("method")[SHOW].round(4))


def hard_case_by_model(results):
    cols = ["Fixed Platt", "Moment matching", "Physics-informed + EM", "SDQC (probe + EM)", "Oracle (noise + true prior)"]
    h = results[(results.hardware_family == HARD) & (results.prior_type == "changed disease rate")]
    return h[h.method.isin(cols)].groupby(["dataset", "model", "method"])["Brier"].mean().unstack("method")[cols].round(4)


def hard_case_by_prior(results):
    cols = ["Fixed Platt", "Moment matching", "EM only", "Physics-informed + EM", "SDQC (probe only)",
            "SDQC (probe + EM)", "Oracle (noise + true prior)"]
    h = results[(results.hardware_family == HARD) & (results.prior_type == "changed disease rate")]
    t = h[h.method.isin(cols)].groupby(["dataset", "target_prior", "method"])["Brier"].mean().unstack("method")[cols].round(4)
    n = h[h.method == "Fixed Platt"].groupby(["dataset", "target_prior"])["n"].first()
    t.insert(0, "n", n)
    return t


def paired_bootstrap(results, n_boot=2000, seed=0):
    """SDQC (probe + EM) minus comparator; cluster bootstrap over seed x model x hardware condition."""
    keys = ["dataset", "seed", "model", "hardware_id", "target_prior"]
    pv = results.pivot_table(index=keys + ["hardware_family", "prior_type"], columns="method", values="Brier").reset_index()
    pv["cluster"] = pv.seed.astype(str) + "|" + pv.model + "|" + pv.hardware_id.astype(str)
    rb = np.random.default_rng(seed)
    rows = []
    for (ds, fam, pt), g in pv[pv.hardware_family != "clean"].groupby(["dataset", "hardware_family", "prior_type"]):
        clusters = g.cluster.unique()
        for comp in _BOOT_ORDER:
            d = (g["SDQC (probe + EM)"] - g[comp]).groupby(g.cluster).mean()
            boots = [d.loc[rb.choice(clusters, len(clusters))].mean() for _ in range(n_boot)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append(dict(dataset=ds, hardware=fam, disease_rate=pt, vs=comp, diff=round(d.mean(), 4),
                             CI95=f"[{lo:.4f}, {hi:.4f}]", ci_low=lo, ci_high=hi,
                             SDQC="better" if hi < 0 else ("worse" if lo > 0 else "no clear difference")))
    return pd.DataFrame(rows)


def prior_error(results):
    est = results[results.method.isin(EST) & (results.prior_type == "changed disease rate")].copy()
    est["abs_error"] = (est.estimated_prior - est.actual_prior).abs()
    return est.groupby(["dataset", "hardware_family", "method"])["abs_error"].mean().unstack().round(3)


def probe_diagnostics(probes):
    p = probes[probes.hardware_family != "clean"]
    return p.groupby(["dataset", "reference"])[["slope", "probe_rmse"]].mean().round(3)


FIG_METHODS = ["Fixed Platt", "Moment matching", "EM only", "Physics-informed + EM", "SDQC (probe + EM)",
               "Oracle (noise + true prior)"]


def hard_case_figure(results, path):
    """Saves <path>.pdf and <path>.png (path may be given with or without an extension)."""
    hard = results[(results.hardware_family == HARD) & (results.prior_type == "changed disease rate")]
    means = {ds: hard[hard.dataset == ds].groupby("method")["Brier"].mean() for ds in hard.dataset.unique()}
    plot_hard_case(means, path)


def plot_hard_case(means, path):
    """means: {dataset: Series(method -> mean Brier)}."""
    FIG = FIG_METHODS
    dsets = list(means)
    if not dsets:
        return
    fig, axes = plt.subplots(1, len(dsets), figsize=(4.6 * len(dsets), 3.8), squeeze=False)
    for ax, ds in zip(axes[0], dsets):
        v = means[ds].reindex(FIG)
        ax.barh(range(len(FIG)), v.values, color=["#9aa5b1"] * 4 + ["#c2410c", "#334155"])
        ax.set_yticks(range(len(FIG))); ax.set_yticklabels(FIG if ax is axes[0][0] else [""] * len(FIG), fontsize=9)
        ax.invert_yaxis(); ax.set_title(ds, fontsize=11); ax.set_xlabel("Brier score (lower is better)")
    fig.suptitle("Unseen noise (amplitude damping) + changed disease rate", fontsize=12)
    plt.tight_layout()
    stem = str(path).rsplit(".", 1)[0] if str(path).endswith((".png", ".pdf")) else str(path)
    for ext in ("pdf", "png"):
        plt.savefig(f"{stem}.{ext}", dpi=200)
    plt.close(fig)


def analyze(results_dir, n_boot=2000):
    out = Path(results_dir)
    results = _prior_type(pd.read_csv(out / "metrics.csv"))
    sanity = pd.read_csv(out / "sanity.csv")
    probes = pd.read_csv(out / "probe_diagnostics.csv")
    tdir = out / "tables"; tdir.mkdir(exist_ok=True)

    tables = {
        "sanity_auroc": sanity.groupby(["dataset", "model"])[["val_AUROC", "test_AUROC", "passed_gate"]].mean().round(3),
        "brier_main": brier_table(results),
        "hard_case_by_model": hard_case_by_model(results),
        "hard_case_by_prior": hard_case_by_prior(results),
        "prior_estimation_error": prior_error(results),
        "probe_diagnostics": probe_diagnostics(probes),
    }
    h = results[(results.hardware_family == HARD) & (results.prior_type == "changed disease rate") & results.method.isin(SHOW)]
    for metric in ["NLL", "ECE"]:
        tables[f"hard_case_{metric}"] = h.groupby(["dataset", "method"])[metric].mean().unstack("method")[SHOW].round(3)
    paired = paired_bootstrap(results, n_boot)
    for pt, tag in [("changed disease rate", "changed"), ("same disease rate", "same")]:
        tables[f"paired_{tag}"] = (paired[paired.disease_rate == pt]
                                   .assign(cell=lambda d: d["diff"].map("{:+.4f}".format) + " " + d.CI95)
                                   .pivot_table(index=["dataset", "hardware"], columns="vs", values="cell", aggfunc="first")
                                   [COMPARATORS])
    paired.to_csv(tdir / "paired_sdqc_long.csv", index=False)

    md = ["# Results summary\n", "Paired differences are SDQC (probe + EM) minus comparator (negative = SDQC better).\n"]
    for name, t in tables.items():
        t.to_csv(tdir / f"{name}.csv")
        md += [f"\n## {name}\n", t.to_markdown() if _has_tabulate() else t.to_string(), "\n"]
    (out / "SUMMARY.md").write_text("\n".join(md))
    hard_case_figure(results, out / "hard_case.png")
    print(f"Wrote {len(tables)} tables to {tdir}, SUMMARY.md and hard_case.pdf/.png")
    return tables


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False
