"""Experiment configuration (identical defaults to the V5.1 notebook used in the paper)."""
import copy

N_QUBITS = 4
N_LAYERS = 3

DEFAULT = dict(
    DATASETS=["pneumoniamnist", "breastmnist", "octmnist"],
    IMG_SIZE=64,
    OCT_MAX_POOL=15000,          # OCTMNIST train+val images kept (stratified); None = all
    TEST_FRAC=0.15, CAL_FRAC=0.15, VAL_FRAC=0.10,
    N_QUBITS=N_QUBITS, N_LAYERS=N_LAYERS,
    EPOCHS=40, BATCH=64, LR=0.02, PATIENCE=8,
    SEEDS=[0, 1, 2],
    QSVM_N_TRAIN=600, QSVM_C=1.0,
    AUROC_GATE=0.75,
    # Hardware drift
    FIT_P=[0.0, 0.002, 0.01, 0.03],
    HELDOUT_P=[0.001, 0.005, 0.015, 0.02, 0.05],
    UNSEEN_AMP=[0.01, 0.03], UNSEEN_BASE_P=0.002,
    N_READOUT_TRAIN=6, N_READOUT_TEST=4, READOUT_MAX=0.08, DRIFT_SHOTS=1000,
    REF=dict(p=0.002, g=0.0, e0=0.02, e1=0.02),     # calibration-day hardware state
    REPORT_ERROR=0.2, COND_SEED=123,
    PRIORS=["source", 0.30, 0.50, 0.90],
    # SDQC
    PROBE_N=96,
    REF_PROBE_SHOTS=10000,
    SDQC_MIN_SLOPE=0.10, SDQC_MAX_SLOPE=5.0,
    SDQC_EM_SHRINK=0.85,
    N_BINS=15, N_BOOT=2000,
    # Models evaluated (quantum models enter the calibration study; classical ones are sanity baselines)
    MODELS=["VQC", "Re-uploading VQC", "QSVM", "MLP", "RBF-SVM"],
)

QUICK = dict(
    DATASETS=["pneumoniamnist", "octmnist"], OCT_MAX_POOL=3000, EPOCHS=5, SEEDS=[0], QSVM_N_TRAIN=150,
    FIT_P=[0.0, 0.002, 0.03], HELDOUT_P=[0.005, 0.05], UNSEEN_AMP=[0.03],
    N_READOUT_TRAIN=3, N_READOUT_TEST=2, N_BOOT=200, QUICK_TRAIN_CAP=600, QUICK_HARDWARE=True,
)


def make_config(quick=False, **overrides):
    """Return a fresh config dict. `quick=True` gives a few-minute smoke-test configuration."""
    cfg = copy.deepcopy(DEFAULT)
    cfg.update(QUICK_TRAIN_CAP=None, QUICK_HARDWARE=False)
    if quick:
        cfg.update(copy.deepcopy(QUICK))
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg
