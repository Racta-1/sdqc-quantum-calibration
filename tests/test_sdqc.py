"""Fast unit tests (no dataset download). Run: pytest -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdqc import SDQC, em_prior_with_info, fit_probe_response, sample_label_shift  # noqa: E402
from sdqc.calibration import prior_adjust, sigmoid  # noqa: E402
from sdqc.circuits import Simulator  # noqa: E402
from sdqc.config import make_config  # noqa: E402
from sdqc.hardware import build_conditions  # noqa: E402


def test_probe_response_recovers_affine_drift():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 2, 96)
    q = 0.6 * r + 0.3
    correct, info = fit_probe_response(r, q)
    assert info["status"] == "ok"
    assert abs(info["slope"] - 0.6) < 1e-8 and abs(info["intercept"] - 0.3) < 1e-8
    np.testing.assert_allclose(correct(q), r, atol=1e-8)


def test_probe_response_degenerate_inputs():
    _, info = fit_probe_response(np.ones(10), np.arange(10.0))
    assert info["status"] == "degenerate"
    _, info = fit_probe_response(np.arange(10.0), -np.arange(10.0))
    assert info["status"] == "nonpositive_slope"


def test_em_recovers_prior_with_calibrated_model():
    rng = np.random.default_rng(1)
    pi_s, pi_t, n = 0.7, 0.3, 20000
    y = (rng.random(n) < pi_t).astype(int)
    z = rng.normal(np.where(y == 1, 1.5, -1.5), 1.0)
    # exact source-calibrated posterior for this Gaussian model
    p_src = prior_adjust(sigmoid(3.0 * z), 0.5, pi_s)
    _, pi_hat = em_prior_with_info(p_src, pi_s, shrink=1.0)
    assert abs(pi_hat - pi_t) < 0.02
    _, pi_shrunk = em_prior_with_info(p_src, pi_s, shrink=0.85)
    assert abs(pi_shrunk - (pi_s + 0.85 * (pi_hat - pi_s))) < 1e-9


def test_label_shift_sampler():
    y = np.r_[np.ones(700), np.zeros(300)].astype(int)
    assert len(sample_label_shift(y, "source", 0)) == 1000
    ids = sample_label_shift(y, 0.5, 0)
    assert abs(y[ids].mean() - 0.5) < 0.01 and len(np.unique(ids)) == len(ids)


def test_sdqc_wrapper_undoes_drift():
    rng = np.random.default_rng(2)
    y = (rng.random(2000) < 0.6).astype(int)
    z = rng.normal(np.where(y == 1, 1.0, -1.0), 1.0)
    probes = rng.normal(0, 1.5, 96)
    cal = SDQC(use_em=False).fit(z, y, probes)
    p_clean, _ = cal.predict(z, probes)
    p_drift, info = cal.predict(0.5 * z - 0.2, 0.5 * probes - 0.2)
    np.testing.assert_allclose(p_clean, p_drift, atol=1e-8)
    assert abs(info["slope"] - 0.5) < 1e-8


def test_conditions_count_matches_paper():
    c = build_conditions(make_config())
    fams = [f for f, _ in c["HARDWARE"]]
    assert len(fams) == 29 and fams.count("held-out depolarizing") == 20 and fams.count("unseen: amplitude damping") == 8


def test_torch_simulator_matches_numpy():
    rng = np.random.default_rng(3)
    X, W = rng.uniform(0, np.pi, (40, 4)), rng.uniform(0, 6, (3, 4, 3))
    sim_np = Simulator("cpu")
    sim_t = Simulator("cpu"); sim_t.use_torch = True       # torch interface on CPU
    from sdqc.circuits import _fm_dm, vqc_circuit
    import pennylane as qml
    dev = qml.device("default.mixed", wires=4)
    sim_t._vqc_noisy = qml.QNode(vqc_circuit, dev, interface="torch", diff_method=None)
    sim_t._dm = qml.QNode(_fm_dm, dev, interface="torch", diff_method=None)
    np.testing.assert_allclose(sim_np.vqc_noisy(X, W, 0.01, 0.02, True), sim_t.vqc_noisy(X, W, 0.01, 0.02, True), atol=1e-10)
    np.testing.assert_allclose(sim_np.density_matrices(X, 0.01, 0.0), sim_t.density_matrices(X, 0.01, 0.0), atol=1e-10)
