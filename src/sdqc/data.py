"""MedMNIST loading, frozen ResNet-18 feature extraction (GPU if available), splits and PCA-4 encoding."""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import N_QUBITS as NQ

SPLITS = ["train", "val", "cal", "test", "shift"]


def resolve_device(device="auto"):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_split(flag, split, img_size):
    import medmnist
    from medmnist import INFO
    DataClass = getattr(medmnist, INFO[flag]["python_class"])
    try:
        ds = DataClass(split=split, download=True, size=img_size)
    except TypeError:
        ds = DataClass(split=split, download=True)
    labels = ds.labels.ravel().astype(int)
    if flag == "octmnist":
        keep = np.isin(labels, [0, 3])
        return ds.imgs[keep], (labels[keep] == 0).astype(int)  # CNV = 1, NORMAL = 0
    return ds.imgs, labels


class FeatureExtractor:
    """Frozen ImageNet ResNet-18 (512-d penultimate features). Runs on GPU when available."""

    def __init__(self, device="auto", batch_size=256):
        self.device = resolve_device(device)
        enc = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        enc.fc = nn.Identity()
        self.encoder = enc.float().eval().to(self.device)
        self.bs = batch_size
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, imgs):
        out = []
        for i in range(0, len(imgs), self.bs):
            x = torch.from_numpy(imgs[i:i + self.bs]).float().to(self.device) / 255.0
            x = x.unsqueeze(1) if x.ndim == 3 else x.permute(0, 3, 1, 2)
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            x = nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
            out.append(self.encoder((x - self.mean) / self.std).float().cpu().numpy())
        return np.concatenate(out).astype(np.float64)


def cache_path(cache_dir, flag, cfg):
    cap = cfg["OCT_MAX_POOL"] if flag == "octmnist" else None
    return os.path.join(cache_dir, f"{flag}_{cfg['IMG_SIZE']}_binaryv5_{cap or 'all'}.npz")


def extract_or_load_features(flag, cfg, cache_dir, device="auto", extractor=None, verbose=True):
    """Return pooled (train+val) features/labels and official-test ('shift') features/labels. Cached as .npz."""
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(cache_dir, flag, cfg)
    if os.path.exists(path):
        d = np.load(path)
        return d["Fp"], d["yp"], d["Fs"], d["ys"]
    t0 = time.time()
    extractor = extractor or FeatureExtractor(device)
    cap = cfg["OCT_MAX_POOL"] if flag == "octmnist" else None
    Xtr, ytr = load_split(flag, "train", cfg["IMG_SIZE"])
    Xva, yva = load_split(flag, "val", cfg["IMG_SIZE"])
    Xte, yte = load_split(flag, "test", cfg["IMG_SIZE"])
    Xp, yp = np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva])
    if cap and len(yp) > cap:
        keep = train_test_split(np.arange(len(yp)), train_size=cap, stratify=yp, random_state=0)[0]
        Xp, yp = Xp[keep], yp[keep]
    Fp, Fs = extractor(Xp), extractor(Xte)
    np.savez(path, Fp=Fp, yp=yp, Fs=Fs, ys=yte)
    if verbose:
        print(f"{flag}: features in {time.time() - t0:.0f}s on {extractor.device}", flush=True)
    return Fp, yp, Fs, yte


def build_dataset(flag, cfg, cache_dir, device="auto", verbose=True):
    """Pooled re-split (train/val/cal/test share the source prevalence); official test kept as 'shift'.
    Features: standardize -> PCA(4) -> standardize -> angles in [0, pi] (no wrap-around)."""
    F_pool, y_pool, F_shift, y_shift = extract_or_load_features(flag, cfg, cache_dir, device, verbose=verbose)
    idx = np.arange(len(y_pool))
    r_, i_test = train_test_split(idx, test_size=cfg["TEST_FRAC"], stratify=y_pool, random_state=0)
    r_, i_cal = train_test_split(r_, test_size=cfg["CAL_FRAC"] / (1 - cfg["TEST_FRAC"]), stratify=y_pool[r_], random_state=0)
    i_tr, i_val = train_test_split(r_, test_size=cfg["VAL_FRAC"] / (1 - cfg["TEST_FRAC"] - cfg["CAL_FRAC"]),
                                   stratify=y_pool[r_], random_state=0)
    if cfg.get("QUICK_TRAIN_CAP"):
        i_tr = i_tr[:cfg["QUICK_TRAIN_CAP"]]
    F = {"train": F_pool[i_tr], "val": F_pool[i_val], "cal": F_pool[i_cal], "test": F_pool[i_test], "shift": F_shift}
    Y = {"train": y_pool[i_tr], "val": y_pool[i_val], "cal": y_pool[i_cal], "test": y_pool[i_test], "shift": y_shift}
    sc1 = StandardScaler().fit(F["train"])
    pca = PCA(n_components=NQ, random_state=0).fit(sc1.transform(F["train"]))
    sc2 = StandardScaler().fit(pca.transform(sc1.transform(F["train"])))
    Z = {k: sc2.transform(pca.transform(sc1.transform(v))) for k, v in F.items()}
    A = {k: np.clip(v, -3, 3) * (np.pi / 6) + np.pi / 2 for k, v in Z.items()}
    if verbose:
        print(f"{flag}: PCA variance kept={pca.explained_variance_ratio_.sum():.2f} | "
              + " ".join(f"{k}={len(Y[k])}({Y[k].mean():.2f})" for k in SPLITS), flush=True)
    return dict(Z=Z, A=A, Y=Y, flag=flag)
