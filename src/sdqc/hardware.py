"""Simulated hardware conditions (calibration-day reference, training grid, held-out and unseen test drift)."""
import zlib

import numpy as np


def cond(p=0.0, g=0.0, e0=0.0, e1=0.0, shots=None):
    return dict(p=p, g=g, e0=e0, e1=e1, shots=shots)


CLEAN = cond()


def build_conditions(cfg):
    """Returns dict with REF, TRAIN_CONDS (for noise-conditioned / physics-informed baselines),
    TEST_CONDS [(family, cond)], and HARDWARE = [("clean", CLEAN)] + TEST_CONDS."""
    rc = np.random.default_rng(cfg["COND_SEED"])
    em, sh = cfg["READOUT_MAX"], cfg["DRIFT_SHOTS"]
    train = [cond(p, 0.0, rc.uniform(0, em), rc.uniform(0, em), sh)
             for p in cfg["FIT_P"] for _ in range(cfg["N_READOUT_TRAIN"])]
    test = [("held-out depolarizing", cond(p, 0.0, rc.uniform(0, em), rc.uniform(0, em), sh))
            for p in cfg["HELDOUT_P"] for _ in range(cfg["N_READOUT_TEST"])]
    test += [("unseen: amplitude damping", cond(cfg["UNSEEN_BASE_P"], gm, rc.uniform(0, em), rc.uniform(0, em), sh))
             for gm in cfg["UNSEEN_AMP"] for _ in range(cfg["N_READOUT_TEST"])]
    hardware = [("clean", CLEAN)] + test
    if cfg.get("QUICK_HARDWARE"):
        hardware = [("clean", CLEAN), test[0], test[-1]]
    ref = cond(**cfg["REF"], shots=sh)
    # `rc` continues to be used for reported rates when the notebook's sequential stream is requested
    return dict(REF=ref, REF_PROBE=dict(ref, shots=cfg["REF_PROBE_SHOTS"]),
                TRAIN_CONDS=train, TEST_CONDS=test, HARDWARE=hardware, rc=rc)


def stable_key(*parts):
    """Deterministic integer from strings/ints (independent of process and run order)."""
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def reported(c, rng, report_error):
    """What the device vendor reports: rates with multiplicative error exp(N(0, sigma)); amplitude damping never reported."""
    f = lambda v: v * np.exp(rng.normal(0, report_error))
    return cond(f(c["p"]), 0.0, f(c["e0"]), f(c["e1"]), c["shots"])
