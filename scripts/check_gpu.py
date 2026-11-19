#!/usr/bin/env python
"""Check that PyTorch sees your GPU and that the GPU simulator matches the CPU simulator."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sdqc.circuits import Simulator  # noqa: E402

print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
rng = np.random.default_rng(0)
X, W = rng.uniform(0, np.pi, (2000, 4)), rng.uniform(0, 2 * np.pi, (3, 4, 3))
t = time.time(); a = Simulator("cpu").vqc_noisy(X, W, 0.01, 0.02, True); tc = time.time() - t
print(f"CPU (NumPy)  : {tc:.2f}s for 2000 noisy circuits")
if torch.cuda.is_available():
    sim = Simulator("cuda"); sim.vqc_noisy(X[:10], W, 0.01, 0.02, True)   # warm-up
    t = time.time(); b = sim.vqc_noisy(X, W, 0.01, 0.02, True); tg = time.time() - t
    print(f"GPU (PyTorch): {tg:.2f}s | max |CPU - GPU| = {np.abs(a - b).max():.1e}")
    print("Use --sim-device cuda" if tg < tc else "CPU is faster here; keep --sim-device cpu (use --jobs instead)")
