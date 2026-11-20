"""Introduction figure: hardware drift and prevalence change are confounded in unlabeled scores; fixed probes separate them.
Illustrative (synthetic Gaussian scores), not experimental data."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42   # embed TrueType (IEEE PDF eXpress rejects Type 3 fonts)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["ps.fonttype"] = 42
import numpy as np
from scipy.stats import norm

INK, MUTED = "#1f2933", "#52606d"
POS, NEG, ALL = "#c2410c", "#0f766e", "#334155"
PROBE = "#7c3aed"
x = np.linspace(-4.5, 4.5, 600)


def mix(pi, mu, sd, shrink=1.0, shift=0.0):
    m1, m0, s = mu * shrink + shift, -mu * shrink + shift, sd
    return pi * norm.pdf(x, m1, s), (1 - pi) * norm.pdf(x, m0, s)


fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.3), gridspec_kw=dict(wspace=0.28))
mu, sd, pi_s = 1.6, 0.8, 0.70

# (a) calibration day
p1, p0 = mix(pi_s, mu, sd)
ax = axes[0]
ax.fill_between(x, p1, color=POS, alpha=.35, lw=0); ax.fill_between(x, p0, color=NEG, alpha=.35, lw=0)
ax.plot(x, p1 + p0, color=ALL, lw=1.8)
ax.set_title("(a) Calibration day\nreference hardware, prevalence 70%", fontsize=10.5, color=INK)
ax.text(2.2, max(p1) * 1.02, "disease", color=POS, fontsize=9, ha="center")
ax.text(-2.2, max(p0) * 1.25, "healthy", color=NEG, fontsize=9, ha="center")

# (b) hardware drift only
p1, p0 = mix(pi_s, mu, sd, shrink=0.5, shift=-0.7)
ax = axes[1]
ax.fill_between(x, p1, color=POS, alpha=.35, lw=0); ax.fill_between(x, p0, color=NEG, alpha=.35, lw=0)
ax.plot(x, p1 + p0, color=ALL, lw=1.8)
ax.set_title("(b) Hardware drift only\nnoisier device, prevalence 70%", fontsize=10.5, color=INK)

# (c) prevalence change only
p1, p0 = mix(0.35, mu, sd)
ax = axes[2]
ax.fill_between(x, p1, color=POS, alpha=.35, lw=0); ax.fill_between(x, p0, color=NEG, alpha=.35, lw=0)
ax.plot(x, p1 + p0, color=ALL, lw=1.8)
ax.set_title("(c) Prevalence change only\nreference hardware, prevalence 35%", fontsize=10.5, color=INK)

ref1, ref0 = mix(pi_s, mu, sd)
for ax in axes[1:3]:
    ax.plot(x, ref1 + ref0, color=MUTED, lw=1.1, ls="--")
axes[1].text(0.02, 0.98, "dashed: calibration day", transform=axes[1].transAxes, fontsize=8.5, color=MUTED, va="top")
for ax in axes[:3]:
    ax.set_ylim(0, 0.42)
for ax in axes[:3]:
    ax.set_yticks([]); ax.set_xticks([]); ax.set_xlabel("model score", fontsize=9, color=MUTED)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_xlim(-4.5, 4.5)
from matplotlib.patches import FancyBboxPatch
y0 = -0.17
fig.canvas.draw()
b1, b2 = axes[1].get_position(), axes[2].get_position()
fig.add_artist(plt.Line2D([b1.x0, b1.x0, b2.x1, b2.x1], [b1.y0 - 0.07, b1.y0 - 0.10, b1.y0 - 0.10, b1.y0 - 0.07],
                          transform=fig.transFigure, color=INK, lw=1))
fig.text((b1.x0 + b2.x1) / 2, b1.y0 - 0.15,
         "Without labels both look like the same kind of change: the confounding that EM-only and moment matching cannot resolve",
         ha="center", va="top", fontsize=9.5, color=INK)

# (d) probes separate the two
ax = axes[3]
rng = np.random.default_rng(3)
r = rng.normal(0, 1.5, 40)
ax.scatter(r, r + rng.normal(0, .08, 40), s=14, color=MUTED, alpha=.8, label="no drift or (c): probes unchanged")
ax.scatter(r, 0.55 * r - 0.45 + rng.normal(0, .12, 40), s=14, color=PROBE, label="drift (b): probes move")
ax.plot([-4, 4], [-4, 4], color=MUTED, lw=1, ls="--")
ax.plot([-4, 4], [0.55 * -4 - .45, 0.55 * 4 - .45], color=PROBE, lw=1.4)
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
ax.set_xlabel("probe score, calibration day", fontsize=9, color=MUTED)
ax.set_ylabel("probe score, today", fontsize=9, color=MUTED)
ax.set_xticks([]); ax.set_yticks([])
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.set_title("(d) Fixed probes tell them apart\nSDQC fits and undoes the drift", fontsize=10.5, color=INK)
ax.legend(fontsize=7.6, loc="upper left", frameon=False)

for ext in ("pdf", "png"):
    plt.savefig(f"docs/figures/sdqc_intro.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print("saved")
