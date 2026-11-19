"""Shift-Disentangled Quantum Calibration (SDQC).

Calibration day (reference hardware, labels available):
  1. fit Platt scaling on labeled calibration scores;
  2. measure K fixed probe inputs on hardware (high shot budget) -> reference scores r_j.
Deployment (today's hardware, no labels):
  3. re-measure the same probes -> q_j; fit q ~= a*r + b (slope clipped);
  4. map patient scores back: z~ = (z - b) / a; apply the frozen Platt calibrator;
  5. estimate the new prevalence with shrunken EM and prior-adjust the probabilities.
"""
import numpy as np

from .calibration import fit_platt, prior_adjust


def select_probe_indices(n, count, seed=31415):
    """Fixed probe set: `count` indices drawn uniformly at random (fixed seed) from a pool of size n."""
    return np.sort(np.random.default_rng(seed).choice(n, size=min(n, count), replace=False))


def fit_probe_response(z_reference, z_today, min_slope=0.10, max_slope=5.0):
    """No labels. Fit z_today ~= slope * z_reference + intercept on the fixed probes and return
    a function that maps today's scores back to the reference scale, plus fit diagnostics."""
    x, y = np.asarray(z_reference, float), np.asarray(z_today, float)
    if len(x) < 3 or np.std(x) < 1e-8:
        return (lambda z: np.asarray(z)), dict(slope=1., intercept=0., probe_rmse=np.nan, status="degenerate")
    slope, intercept = np.polyfit(x, y, 1)
    if not np.isfinite(slope) or slope <= 0:
        return (lambda z: np.asarray(z)), dict(slope=float(slope), intercept=float(intercept),
                                               probe_rmse=np.nan, status="nonpositive_slope")
    slope = float(np.clip(slope, min_slope, max_slope))
    intercept = float(np.mean(y - slope * x))
    rmse = float(np.sqrt(np.mean((y - (slope * x + intercept)) ** 2)))

    def correct(z):
        return (np.asarray(z, float) - intercept) / slope
    return correct, dict(slope=slope, intercept=intercept, probe_rmse=rmse, status="ok")


def em_prior_with_info(p_source, source_prior, shrink=1., max_iters=200):
    """Estimate the target prevalence from unlabeled probabilities (EM), shrink toward the source
    prevalence, and return (adjusted probabilities, estimated prevalence)."""
    p = np.clip(np.asarray(p_source, float), 1e-6, 1 - 1e-6)
    pi_s = float(np.clip(source_prior, 1e-4, 1 - 1e-4))
    pi = pi_s
    for _ in range(max_iters):
        q = prior_adjust(p, pi_s, pi)
        updated = float(np.clip(q.mean(), 0.01, 0.99))
        if abs(updated - pi) < 1e-7:
            break
        pi = updated
    pi_est = float(pi_s + shrink * (pi - pi_s))
    return prior_adjust(p, pi_s, pi_est), pi_est


def sample_label_shift(y, target_prior, seed):
    """Subsample (without replacement) an evaluation set with the requested positive share."""
    r = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if target_prior == "source":
        return np.arange(len(y))
    pi = float(target_prior)
    n = int(min(len(y), len(pos) / pi, len(neg) / (1 - pi)))
    k = int(round(n * pi))
    if n < 20 or k < 5 or n - k < 5:
        return np.array([], dtype=int)
    return np.sort(np.r_[r.choice(pos, k, replace=False), r.choice(neg, n - k, replace=False)])


class SDQC:
    """Convenience wrapper for using SDQC with your own model / hardware.

    >>> cal = SDQC().fit(z_cal, y_cal, probe_reference_scores)          # calibration day
    >>> p, info = cal.predict(z_today, probe_today_scores)              # deployment, no labels
    """

    def __init__(self, min_slope=0.10, max_slope=5.0, shrink=0.85, use_em=True):
        self.min_slope, self.max_slope, self.shrink, self.use_em = min_slope, max_slope, shrink, use_em

    def fit(self, z_cal, y_cal, probe_reference):
        self.platt = fit_platt(np.asarray(z_cal, float), np.asarray(y_cal, float))
        self.source_prior = float(np.mean(y_cal))
        self.probe_reference = np.asarray(probe_reference, float)
        return self

    def predict(self, z_today, probe_today):
        correct, info = fit_probe_response(self.probe_reference, probe_today, self.min_slope, self.max_slope)
        p = self.platt(correct(z_today))
        if self.use_em:
            p, info["estimated_prior"] = em_prior_with_info(p, self.source_prior, self.shrink)
        return p, info
