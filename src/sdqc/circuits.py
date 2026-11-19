"""Quantum circuits, noise channels and a simulator that can run noisy circuits on CPU (NumPy) or GPU (PyTorch).

All circuits are 4-qubit. Noise is applied after every gate:
  depolarizing(p) and amplitude damping(gamma); readout error and shot noise are applied analytically.
"""
import numpy as np
import pennylane as qml
import torch

from .config import N_QUBITS as NQ, N_LAYERS as NL


def _noise(w, p, g):
    if p > 0:
        qml.DepolarizingChannel(p, wires=w)
    if g > 0:
        qml.AmplitudeDamping(g, wires=w)


def vqc_circuit(x, W, p=0.0, g=0.0, reupload=False):
    """VQC: RY angle encoding (inputs already in [0, pi]), Rot layers, CNOT ring, readout <Z0>."""
    for l in range(NL):
        if l == 0 or reupload:
            for q in range(NQ):
                qml.RY(x[..., q], wires=q); _noise(q, p, g)
        for q in range(NQ):
            qml.Rot(W[l, q, 0], W[l, q, 1], W[l, q, 2], wires=q); _noise(q, p, g)
        for q in range(NQ):
            t = (q + 1) % NQ
            qml.CNOT(wires=[q, t]); _noise(q, p, g); _noise(t, p, g)
    return qml.expval(qml.PauliZ(0))


def feature_map(x, p=0.0, g=0.0):
    """QSVM fidelity-kernel feature map: 2 repetitions of RY/RZ + CNOT chain."""
    for _ in range(2):
        for q in range(NQ):
            qml.RY(x[..., q], wires=q); _noise(q, p, g)
            qml.RZ(x[..., q], wires=q); _noise(q, p, g)
        for q in range(NQ - 1):
            qml.CNOT(wires=[q, q + 1]); _noise(q, p, g); _noise(q + 1, p, g)


def _fm_state(x):
    feature_map(x)
    return qml.state()


def _fm_dm(x, p, g):
    feature_map(x, p, g)
    return qml.density_matrix(wires=range(NQ))


def batched(fn, X, *args):
    """Try parameter broadcasting first; fall back to a Python loop."""
    try:
        return np.asarray(fn(X, *args))
    except Exception:
        return np.stack([np.asarray(fn(x, *args)) for x in X])


def apply_readout(f, e0, e1):
    """Asymmetric readout error on an expectation value <Z>."""
    return f * (1 - e0 - e1) + (e1 - e0)


def apply_shots_expval(f, shots, rng):
    """Finite-shot estimate of <Z> by binomial sampling."""
    return 1 - 2 * rng.binomial(shots, np.clip((1 - f) / 2, 0, 1)) / shots


class Simulator:
    """Holds PennyLane devices/QNodes.

    sim_device="cpu"  -> noisy density-matrix circuits use the NumPy interface (as in the paper).
    sim_device="cuda" -> noisy circuits run through the PyTorch interface on the GPU, in chunks.
    Both give numerically identical results (max abs difference ~1e-16 in our checks).
    Training of the VQC (noise-free, 38 parameters) always runs on CPU: it is too small to benefit from a GPU.
    """

    def __init__(self, sim_device="cpu", chunk=4096):
        self.sim_device = sim_device
        self.use_torch = sim_device != "cpu"
        self.tdev = torch.device(sim_device) if self.use_torch else None
        self.chunk = chunk
        dev_ideal = qml.device("default.qubit", wires=NQ)
        dev_mixed = qml.device("default.mixed", wires=NQ)
        self.vqc_train = qml.QNode(vqc_circuit, dev_ideal, interface="torch", diff_method="backprop")
        self.state = qml.QNode(_fm_state, dev_ideal, interface="numpy")
        if self.use_torch:
            self._vqc_noisy = qml.QNode(vqc_circuit, dev_mixed, interface="torch", diff_method=None)
            self._dm = qml.QNode(_fm_dm, dev_mixed, interface="torch", diff_method=None)
        else:
            self._vqc_noisy = qml.QNode(vqc_circuit, dev_mixed, interface="numpy")
            self._dm = qml.QNode(_fm_dm, dev_mixed, interface="numpy")

    def _t(self, a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float64, device=self.tdev)

    def vqc_noisy(self, X, W, p, g, reupload):
        """Noise-channel expectation <Z0> for every row of X (no readout / shot noise)."""
        if not self.use_torch:
            return batched(lambda Xb: self._vqc_noisy(Xb, W, p, g, reupload), X).astype(float)
        Wt = self._t(W)
        out = []
        with torch.no_grad():
            for i in range(0, len(X), self.chunk):
                out.append(self._vqc_noisy(self._t(X[i:i + self.chunk]), Wt, p, g, reupload).detach().cpu().numpy())
        return np.concatenate(out).astype(float)

    def density_matrices(self, X, p, g):
        """Noisy feature-map density matrices, flattened to (n, 256)."""
        if not self.use_torch:
            return batched(self._dm, X, p, g).reshape(len(X), -1)
        out = []
        with torch.no_grad():
            for i in range(0, len(X), self.chunk):
                out.append(self._dm(self._t(X[i:i + self.chunk]), p, g).detach().cpu().numpy())
        return np.concatenate(out).reshape(len(X), -1)

    def states(self, X):
        return batched(self.state, X)
