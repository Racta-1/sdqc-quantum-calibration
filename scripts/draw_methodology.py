"""Draw the SDQC methodology figure (docs/figures/sdqc_methodology.png). Run from the repo root."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42   # embed TrueType (IEEE PDF eXpress rejects Type 3 fonts)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["ps.fonttype"] = 42
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED = "#1f2933", "#52606d"
PROBE, PROBE_BG = "#0f766e", "#e6f4f1"     # hardware / probe path
PAT, PAT_BG = "#c2410c", "#fdf0e8"         # patient path
STORE_BG, STORE_EC = "#eef1f4", "#7b8794"
LANE = "#f7f8fa"

fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 130); ax.set_ylim(0, 72); ax.axis("off")

def lane(y0, h, title, sub):
    ax.add_patch(FancyBboxPatch((1, y0), 128, h, boxstyle="round,pad=0,rounding_size=1.5", fc=LANE, ec="#d5dae0", lw=1))
    ax.text(3, y0 + h - 2.2, title, fontsize=12.5, weight="bold", color=INK, va="top")
    ax.text(3, y0 + h - 5.4, sub, fontsize=9.5, color=MUTED, va="top")

def box(x, y, w, h, text, fc, ec, size=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.35)

def arrow(x1, y1, x2, y2, color=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, color=color,
                                 lw=1.5, shrinkA=1, shrinkB=1))

# ---------------- lanes ----------------
lane(39, 31, "Calibration day", "Hardware in reference state H₀  ·  labeled calibration data available  ·  done once")
lane(1, 36, "Deployment", "Hardware has drifted to Hₜ  ·  new patient population  ·  no new labels")

# ---------------- calibration day ----------------
y1, y2, h = 53, 42, 7.5
box(4, y1, 21, h, "Labeled calibration\nimages", PAT_BG, PAT)
box(31, y1, 22, h, "Quantum classifier\non H₀", "white", INK)
box(59, y1, 24, h, "Fit Platt calibrator g\n(score → probability)", "white", INK)
box(91, y1, 34, h, "Stored: calibrator g\nand source disease rate πₛ", STORE_BG, STORE_EC)
for a, b in [(25, 31), (53, 59), (83, 91)]:
    arrow(a, y1 + h / 2, b, y1 + h / 2)

box(4, y2, 21, h, "Fixed probe images\n(labels never used)", PROBE_BG, PROBE)
box(31, y2, 22, h, "Quantum classifier\non H₀, many shots", "white", PROBE)
box(91, y2, 34, h, "Stored: reference probe\nscores r₁ … rₙ", STORE_BG, STORE_EC)
arrow(25, y2 + h / 2, 31, y2 + h / 2, PROBE)
arrow(53, y2 + h / 2, 91, y2 + h / 2, PROBE)

# ---------------- deployment ----------------
yp, yz, hh = 17.5, 4, 8.5
box(4, yp, 21, hh, "Same probe images", PROBE_BG, PROBE)
box(31, yp, 22, hh, "Quantum classifier\non Hₜ", "white", PROBE)
box(59, yp, 34, hh, "Step 1a. Fit hardware map\nqᵢ ≈ a·rᵢ + b   (rᵢ stored)", "white", PROBE)
arrow(25, yp + hh / 2, 31, yp + hh / 2, PROBE)
arrow(53, yp + hh / 2, 59, yp + hh / 2, PROBE)
ax.text(56, yp + hh / 2 + 1.3, "qᵢ", fontsize=9.5, color=PROBE, ha="center")
ax.text(97, yp + hh / 2, "Probes are the same images\nevery time, so their scores\nchange only when the\nhardware changes.",
        fontsize=9, color=PROBE, va="center", ha="left", style="italic", linespacing=1.35)

box(4, yz, 21, hh, "New patient images\n(unlabeled)", PAT_BG, PAT)
box(31, yz, 22, hh, "Quantum classifier\non Hₜ", "white", PAT)
box(59, yz, 16, hh, "Step 1b.\nz′ = (z − b) / a", "white", PAT)
box(80, yz, 17, hh, "Step 2.\np = g(z′)   (g stored)", "white", PAT, size=9)
box(102, yz, 23, hh, "Step 3. EM re-estimates\nthe disease rate and\nadjusts p → risk", "white", PAT, size=9)
arrow(25, yz + hh / 2, 31, yz + hh / 2, PAT)
arrow(53, yz + hh / 2, 59, yz + hh / 2, PAT)
ax.text(56, yz + hh / 2 + 1.3, "z", fontsize=9.5, color=PAT, ha="center")
arrow(75, yz + hh / 2, 80, yz + hh / 2, PAT)
arrow(97, yz + hh / 2, 102, yz + hh / 2, PAT)
arrow(67, yp, 67, yz + hh, PROBE)
ax.text(68.2, (yp + yz + hh) / 2, "a, b", fontsize=9, color=PROBE, va="center")

for ext in ("pdf", "png"):
    plt.savefig(f"docs/figures/sdqc_methodology.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print("saved")
