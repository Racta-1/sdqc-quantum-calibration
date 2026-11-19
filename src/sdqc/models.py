"""Quantum and classical classifiers. Every model exposes clean(split) and observe(split, condition, rng)."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from .circuits import apply_readout, apply_shots_expval
from .config import N_LAYERS as NL, N_QUBITS as NQ

torch.set_default_dtype(torch.float64)
EVAL_SPLITS = ["val", "cal", "test", "shift"]


def train_torch(model, Xtr, ytr, Xva, yva, seed, cfg):
    """Adam + BCE with early stopping on validation loss."""
    torch.manual_seed(seed)
    Xtr, ytr = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float64)
    Xva, yva = torch.tensor(Xva), torch.tensor(yva, dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["LR"])
    lossf = nn.BCEWithLogitsLoss()
    best, state, bad, ep = np.inf, None, 0, 0
    for ep in range(cfg["EPOCHS"]):
        perm = torch.randperm(len(ytr))
        for i in range(0, len(ytr), cfg["BATCH"]):
            j = perm[i:i + cfg["BATCH"]]
            opt.zero_grad(); lossf(model(Xtr[j]), ytr[j]).backward(); opt.step()
        with torch.no_grad():
            vl = lossf(model(Xva), yva).item()
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["PATIENCE"]:
                break
    model.load_state_dict(state)
    return model, best, ep + 1


class VQCNet(nn.Module):
    def __init__(self, sim, seed, reupload):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.sim = sim
        self.W = nn.Parameter(torch.rand(NL, NQ, 3, generator=g) * 2 * np.pi)
        self.a = nn.Parameter(torch.tensor(2.0)); self.b = nn.Parameter(torch.tensor(0.0))
        self.reupload = reupload

    def forward(self, x):
        return self.a * self.sim.vqc_train(x, self.W, 0.0, 0.0, self.reupload) + self.b


class VQCModel:
    """VQC (reupload=False) or data re-uploading VQC (reupload=True); logit = a*<Z0> + b."""
    kind = "quantum"

    def __init__(self, name, D, seed, reupload, sim, cfg):
        self.name, self.D, self.reupload, self.sim = name, D, reupload, sim
        net, self.val_loss, self.epochs = train_torch(VQCNet(sim, seed, reupload), D["A"]["train"], D["Y"]["train"],
                                                      D["A"]["val"], D["Y"]["val"], seed, cfg)
        self.W = net.W.detach().numpy(); self.a, self.b = net.a.item(), net.b.item()
        with torch.no_grad():
            self._f = {(s, 0.0, 0.0): sim.vqc_train(torch.tensor(D["A"][s]), net.W, 0.0, 0.0, reupload).detach().numpy()
                       for s in EVAL_SPLITS}

    def exact(self, split, p, g):
        key = (split, p, g)
        if key not in self._f:
            self._f[key] = self.sim.vqc_noisy(self.D["A"][split], self.W, p, g, self.reupload)
        return self._f[key]

    def clean(self, split):
        return self.a * self.exact(split, 0.0, 0.0) + self.b

    def observe(self, split, c, rng):
        f = self.exact(split, c["p"], c["g"])
        if c["e0"] or c["e1"]:
            f = apply_readout(f, c["e0"], c["e1"])
        if c["shots"]:
            f = apply_shots_expval(f, c["shots"], rng)
        return self.a * f + self.b


class QSVMModel:
    """Fidelity-kernel SVM. Noisy kernel = Tr(rho_i rho_j) from noisy density matrices (proxy), shot noise binomial."""
    kind = "quantum"

    def __init__(self, name, D, seed, sim, cfg):
        self.name, self.D, self.sim = name, D, sim
        Ytr = D["Y"]["train"]
        n = min(cfg["QSVM_N_TRAIN"], len(Ytr))
        self.idx = (train_test_split(np.arange(len(Ytr)), train_size=n, stratify=Ytr, random_state=seed)[0]
                    if n < len(Ytr) else np.arange(len(Ytr)))
        self.Xtr = D["A"]["train"][self.idx]
        self._sv = {"train": sim.states(self.Xtr)}
        for s in EVAL_SPLITS:
            self._sv[s] = sim.states(D["A"][s])
        K = np.abs(self._sv["train"].conj() @ self._sv["train"].T) ** 2
        self.clf = SVC(C=cfg["QSVM_C"], kernel="precomputed").fit(K, Ytr[self.idx])
        self._dm = {}

    def _dms(self, split, p, g):
        key = (split, p, g)
        if key not in self._dm:
            X = self.Xtr if split == "train" else self.D["A"][split]
            self._dm[key] = self.sim.density_matrices(X, p, g)
        return self._dm[key]

    def kernel(self, split, p, g):
        if p == 0 and g == 0:
            K = np.abs(self._sv[split].conj() @ self._sv["train"].T) ** 2
        else:
            K = np.real(self._dms(split, p, g) @ self._dms("train", p, g).conj().T)
        return np.clip(np.nan_to_num(K), 0.0, 1.0)      # rounding can give 1.0000000000000002

    def clean(self, split):
        return self.clf.decision_function(self.kernel(split, 0.0, 0.0))

    def observe(self, split, c, rng):
        K = self.kernel(split, c["p"], c["g"])          # readout error not modelled for the kernel proxy
        if c["shots"]:
            K = rng.binomial(c["shots"], K) / c["shots"]
        return self.clf.decision_function(K)


class ClassicalModel:
    """MLP (4-6-1) or RBF-SVM on the same PCA features. Not affected by quantum noise."""
    kind = "classical"

    def __init__(self, name, D, seed, cfg):
        self.name, self.D = name, D
        if name == "MLP":
            torch.manual_seed(seed)
            net = nn.Sequential(nn.Linear(NQ, 6), nn.Tanh(), nn.Linear(6, 1), nn.Flatten(0))
            net, self.val_loss, self.epochs = train_torch(net, D["Z"]["train"], D["Y"]["train"],
                                                          D["Z"]["val"], D["Y"]["val"], seed, cfg)
            with torch.no_grad():
                self._z = {s: net(torch.tensor(D["Z"][s])).numpy() for s in EVAL_SPLITS}
        else:
            svm = SVC(C=1.0, kernel="rbf", gamma="scale", random_state=seed).fit(D["Z"]["train"], D["Y"]["train"])
            self._z = {s: svm.decision_function(D["Z"][s]) for s in EVAL_SPLITS}

    def clean(self, split):
        return self._z[split]

    def observe(self, split, c, rng):
        return self._z[split]


class ControlModel:
    """Classical MLP whose logits receive the same stretch, shift and noise as the VQC's logits (V4 control)."""
    kind = "control"

    def __init__(self, name, mlp, vqc):
        self.name, self.mlp, self.vqc = name, mlp, vqc

    def clean(self, split):
        return self.mlp.clean(split)

    def observe(self, split, c, rng):
        zq = self.vqc.observe(split, c, rng); z0 = self.vqc.clean(split)
        s, off = np.polyfit(z0, zq, 1)
        sd = np.std(zq - (s * z0 + off))
        return s * self.mlp.clean(split) + off + rng.normal(0, sd, len(z0))


def build_models(D, seed, sim, cfg):
    """Instantiate (train) the models listed in cfg['MODELS'], in the paper's order."""
    makers = {
        "VQC": lambda: VQCModel("VQC", D, seed, False, sim, cfg),
        "Re-uploading VQC": lambda: VQCModel("Re-uploading VQC", D, seed, True, sim, cfg),
        "QSVM": lambda: QSVMModel("QSVM", D, seed, sim, cfg),
        "MLP": lambda: ClassicalModel("MLP", D, seed, cfg),
        "RBF-SVM": lambda: ClassicalModel("RBF-SVM", D, seed, cfg),
    }
    return {name: makers[name]() for name in cfg["MODELS"]}
