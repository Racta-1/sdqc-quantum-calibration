"""Main V5.1 benchmark: one (dataset, seed) unit of work -> metric rows, sanity rows, probe diagnostics."""
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from .calibration import fit_device_response, fit_noise_conditioned_platt, fit_platt, metrics, moment_match, prior_adjust
from .hardware import reported, stable_key
from .models import build_models
from .sdqc import em_prior_with_info, fit_probe_response, sample_label_shift, select_probe_indices

# method -> key of the estimated prevalence that goes with it
EST = {"SDQC (probe + EM)": "sdqc", "SDQC-sim (probe + EM)": "sdqc_sim", "EM only": "em", "Physics-informed + EM": "phys"}


def run_dataset_seed(flag, seed, D, sim, cfg, conds, verbose=True, legacy_report_stream=False):
    """legacy_report_stream=True draws the vendor-reported error rates from one sequential stream
    (conds['rc']), exactly as the original notebook; requires running all units in order in one process.
    Default: an independent stream per (dataset, seed, model, hardware condition), so units can run in parallel."""
    t0 = time.time()
    Y = D["Y"]
    source_prior = float(Y["cal"].mean())
    REF, REF_PROBE, TRAIN_CONDS, HARDWARE = conds["REF"], conds["REF_PROBE"], conds["TRAIN_CONDS"], conds["HARDWARE"]
    rows, sanity, probes = [], [], []
    sh = cfg["SDQC_EM_SHRINK"]
    nb = cfg["N_BINS"]

    models = build_models(D, seed, sim, cfg)
    for name, m in models.items():
        val_auc = float(roc_auc_score(Y["val"], m.clean("val")))
        sanity.append(dict(dataset=flag, seed=seed, model=name, val_AUROC=val_auc,
                           test_AUROC=float(roc_auc_score(Y["test"], m.clean("test"))),
                           passed_gate=val_auc >= cfg["AUROC_GATE"]))
        if m.kind != "quantum" or val_auc < cfg["AUROC_GATE"]:
            continue
        rng = np.random.default_rng(100000 + 1000 * seed + sum(map(ord, name)))

        # ---------------- calibration day: hardware in reference state REF ----------------
        fixed = fit_platt(m.observe("cal", REF, rng), Y["cal"])            # calibration-day Platt (labels)
        z_ref = m.observe("cal", REF, rng)                                  # unlabeled stats (moment matching)
        probe_idx = select_probe_indices(len(Y["val"]), cfg["PROBE_N"], 3000 + seed)
        probe_ref_hw = m.observe("val", REF_PROBE, rng)[probe_idx]          # stored hardware reference (no labels)

        # ---------------- comparators that need a noise-free simulator ----------------
        clean_platt = fit_platt(m.clean("cal"), Y["cal"])
        probe_ref_sim = m.clean("val")[probe_idx]
        z_train = [m.observe("cal", c, rng) for c in TRAIN_CONDS]
        noise_cond = fit_noise_conditioned_platt(z_train, TRAIN_CONDS, Y["cal"])
        phys_corr, _ = fit_device_response(m.clean("cal"), z_train, TRAIN_CONDS)

        for hi, (family, hardware) in enumerate(HARDWARE):
            rep_rng = (conds["rc"] if legacy_report_stream
                       else np.random.default_rng(stable_key(cfg["COND_SEED"], flag, seed, name, hi)))
            reported_hw = reported(hardware, rep_rng, cfg["REPORT_ERROR"])
            probe_today = m.observe("val", hardware, rng)[probe_idx]        # same probes, today's hardware
            corr_hw, info_hw = fit_probe_response(probe_ref_hw, probe_today, cfg["SDQC_MIN_SLOPE"], cfg["SDQC_MAX_SLOPE"])
            corr_sim, info_sim = fit_probe_response(probe_ref_sim, probe_today, cfg["SDQC_MIN_SLOPE"], cfg["SDQC_MAX_SLOPE"])
            for ref_name, info in [("hardware (calibration day)", info_hw), ("simulator", info_sim)]:
                probes.append(dict(dataset=flag, seed=seed, model=name, hardware_id=hi,
                                   hardware_family=family, reference=ref_name, **info))
            z_target = m.observe("test", hardware, rng)
            oracle = fit_platt(m.observe("cal", hardware, rng), Y["cal"])
            for prior in cfg["PRIORS"]:
                ids = sample_label_shift(Y["test"], prior, 12345 + hi + seed)
                if not len(ids):
                    continue
                z, y = z_target[ids], Y["test"][ids]
                pi_true = float(y.mean())
                p_fixed = fixed(z)
                p_probe = fixed(corr_hw(z))
                p_sdqc, est_sdqc = em_prior_with_info(p_probe, source_prior, sh)
                p_probe_sim = clean_platt(corr_sim(z))
                p_sdqc_sim, est_sim = em_prior_with_info(p_probe_sim, source_prior, sh)
                p_phys = clean_platt(phys_corr(z, reported_hw))
                p_phys_em, est_phys = em_prior_with_info(p_phys, source_prior, sh)
                p_em_only, est_em = em_prior_with_info(p_fixed, source_prior, sh)
                p_oracle = oracle(z)
                est = dict(sdqc=est_sdqc, sdqc_sim=est_sim, em=est_em, phys=est_phys)
                predictions = {
                    "Fixed Platt": p_fixed,
                    "Oracle Platt (noise only)": p_oracle,
                    "Oracle (noise + true prior)": prior_adjust(p_oracle, source_prior, pi_true),
                    "Noise-conditioned Platt": noise_cond(z, reported_hw),
                    "Physics-informed": p_phys,
                    "Physics-informed + EM": p_phys_em,
                    "Moment matching": fixed(moment_match(z, z_ref.mean(), z_ref.std())),
                    "EM only": p_em_only,
                    "SDQC (probe only)": p_probe,
                    "SDQC (probe + EM)": p_sdqc,
                    "SDQC-sim (probe only)": p_probe_sim,
                    "SDQC-sim (probe + EM)": p_sdqc_sim,
                }
                for method, p in predictions.items():
                    rows.append(dict(dataset=flag, seed=seed, model=name, hardware_id=hi,
                                     hardware_family=family, p=hardware["p"], g=hardware["g"],
                                     e0=hardware["e0"], e1=hardware["e1"],
                                     target_prior=str(prior), actual_prior=pi_true,
                                     estimated_prior=est.get(EST.get(method), np.nan),
                                     method=method, **metrics(p, y, nb)))
    if verbose:
        print(f"{flag} seed {seed}: done in {time.time() - t0:.0f}s", flush=True)
    return rows, sanity, probes
