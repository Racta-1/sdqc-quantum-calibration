"""Calibration metrics and baseline calibrators (Platt, temperature, EM prior correction,
noise-conditioned Platt, physics-informed device response, moment matching)."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import roc_auc_score


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def nll_logits(z, y):
    return np.mean(np.logaddexp(0, z) - y * z)


def ece_score(p, y, n_bins=15):
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return sum((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean()) for k in range(n_bins) if (b == k).any())


def metrics(p, y, n_bins=15):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(Brier=np.mean((p - y) ** 2), NLL=-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)),
                ECE=ece_score(p, y, n_bins), AUROC=roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan,
                Acc=np.mean((p >= 0.5) == y), CITL=p.mean() - y.mean(), n=len(y))


def fit_platt(z, y):
    a_, b_ = minimize(lambda v: nll_logits(v[0] * z + v[1], y), x0=[1.0, 0.0], method="BFGS").x
    return lambda zz: sigmoid(a_ * zz + b_)


def fit_temperature(z, y):
    T = np.exp(minimize_scalar(lambda lt: nll_logits(z / np.exp(lt), y), bounds=(-6, 6), method="bounded").x)
    return lambda zz: sigmoid(zz / T)


def prior_adjust(p, pi_s, pi_t):
    """Bayes-rule adjustment of probabilities from source prior pi_s to target prior pi_t."""
    p = np.clip(p, 1e-6, 1 - 1e-6); num = pi_t / pi_s * p
    return num / (num + (1 - pi_t) / (1 - pi_s) * (1 - p))


def em_prior(p, pi_s, iters=200):
    """Saerens et al. (2002) EM prior re-estimation (no shrinkage)."""
    p = np.clip(p, 1e-6, 1 - 1e-6); pi_t = pi_s; q = p
    for _ in range(iters):
        num = pi_t / pi_s * p; q = num / (num + (1 - pi_t) / (1 - pi_s) * (1 - p))
        new = float(np.clip(q.mean(), 0.01, 0.99))
        if abs(new - pi_t) < 1e-7:
            break
        pi_t = new
    return q


def _eta(c):
    return np.array([c["p"], c["e0"], c["e1"]])


def fit_noise_conditioned_platt(zs, cs, y):
    """Platt whose slope/intercept depend linearly on the reported noise rates."""
    raw = np.array([_eta(c) for c in cs]); mu, sd = raw.mean(0), raw.std(0) + 1e-12
    Phi = np.vstack([np.repeat(((_eta(c) - mu) / sd)[None], len(z), 0) for z, c in zip(zs, cs)])
    zz, yy = np.concatenate(zs), np.tile(y, len(zs))
    w = minimize(lambda w: nll_logits((w[0] + Phi @ w[1:4]) * zz + w[4] + Phi @ w[5:8], yy),
                 x0=np.r_[1.0, 0, 0, 0, 0.0, 0, 0, 0], method="BFGS").x

    def predict(z, c):
        ph = (_eta(c) - mu) / sd
        return sigmoid((w[0] + ph @ w[1:4]) * z + w[4] + ph @ w[5:8])
    return predict


def fit_device_response(z0, zs, cs):
    """Physics-informed: score_obs ~= s(eta)*score_clean + c(eta), with
    log s = k*log(1-4p/3) + m*log(1-e0-e1) and c = c2*(1-s) + c1*(e1-e0).
    Fitted on unlabeled calibration images (paired noise-free vs noisy scores) at the training conditions."""
    S, O = [], []
    for z in zs:
        s, o = np.polyfit(z0, z, 1); S.append(max(s, 1e-4)); O.append(o)
    S, O = np.array(S), np.array(O)
    X1 = np.array([[np.log(1 - 4 * c["p"] / 3), np.log(1 - c["e0"] - c["e1"])] for c in cs])
    k, m = np.linalg.lstsq(X1, np.log(S), rcond=None)[0]
    X2 = np.column_stack([1 - S, [c["e1"] - c["e0"] for c in cs]])
    c2, c1 = np.linalg.lstsq(X2, O, rcond=None)[0]

    def correct(z, c):
        s = np.exp(k * np.log(max(1 - 4 * c["p"] / 3, 1e-6)) + m * np.log(max(1 - c["e0"] - c["e1"], 1e-6)))
        s = max(s, 0.02)
        return (z - (c2 * (1 - s) + c1 * (c["e1"] - c["e0"]))) / s
    return correct, dict(k=k, m=m, c1=c1, c2=c2)


def moment_match(z, ref_mean, ref_std):
    return (z - z.mean()) / (z.std() + 1e-12) * ref_std + ref_mean
