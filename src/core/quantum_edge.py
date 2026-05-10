"""
quantum_edge.py — HQEHED-AMT Quantum Edge Detection pipeline.

Implements the method from Rengasamy et al. (2025):
  - QPIE encoding  (Eq. 2.4)
  - Hadamard + Unitary Decremental edge amplitude  (Eq. 2.8–2.16)
  - Adaptive Mean Thresholding (AMT)  (Eq. 2.17–2.20)
  - PPQEC post-quantum error correction  (Section 2.8.3)
"""

from __future__ import annotations
import math
from typing import List, Optional

import numpy as np
import cv2

from .config import Config

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    _QISKIT_AVAILABLE = True
except ImportError:
    _QISKIT_AVAILABLE = False


def _next_pow2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


class QPIEEncoder:
    """Quantum Probability Image Encoding via Qiskit Statevector (Eq. 2.4).

    Encodes pixel values as quantum amplitudes:
        |psi> = sum_i c_i |i>  where  c_i = I_i / ||I||_2
    """

    def encode(self, row: np.ndarray) -> np.ndarray:
        """Encode a 1-D pixel array as QPIE amplitudes.

        Returns Re(sv)[:N] — equivalent to L2 normalization.
        """
        if not _QISKIT_AVAILABLE:
            raise ImportError("Qiskit is not installed. Run: pip install qiskit qiskit-aer")

        N = len(row)
        N_pad = _next_pow2(N)
        n_q = int(math.log2(N_pad))

        arr = np.zeros(N_pad, dtype=np.float64)
        arr[:N] = row.astype(np.float64)
        norm = np.linalg.norm(arr)
        if norm < 1e-10:
            return np.zeros(N, dtype=np.float32)
        arr /= norm

        qc = QuantumCircuit(n_q)
        qc.initialize(arr.tolist(), list(range(n_q)))
        sv = Statevector(qc)
        return np.real(sv.data).astype(np.float32)[:N]


class HadamardEdgeAmplitude:
    """Quantum Hadamard + Unitary Decremental edge amplitude (Eq. 2.8–2.16).

    Computes forward cyclic difference on QPIE-encoded amplitudes:
        diff[i] = sv[i] - sv[(i+1) % N_pad]
    """

    def __init__(self):
        self._encoder = QPIEEncoder()

    def compute(self, p: np.ndarray) -> np.ndarray:
        N = len(p)
        N_pad = _next_pow2(N)

        arr = np.zeros(N_pad, dtype=np.float64)
        arr[:N] = p.astype(np.float64)
        norm = np.linalg.norm(arr)
        if norm < 1e-10:
            return np.zeros(N, dtype=np.float32)
        arr /= norm

        if not _QISKIT_AVAILABLE:
            raise ImportError("Qiskit is not installed.")

        qc = QuantumCircuit(int(math.log2(N_pad)))
        qc.initialize(arr.tolist(), list(range(int(math.log2(N_pad)))))
        sv = Statevector(qc)
        sv_real = np.real(sv.data).astype(np.float64)

        diff = np.empty(N_pad, dtype=np.float32)
        for i in range(N_pad):
            diff[i] = float(sv_real[i] - sv_real[(i + 1) % N_pad])
        if N < N_pad:
            diff[N - 1] = 0.0
        return diff[:N]


class AdaptiveMeanThreshold:
    """QSVM Adaptive Mean Thresholding (Eq. 2.17–2.20).

        mu  = mean(|diff|)
        T   =  gamma     * mu   (positive edge threshold)
        T_  = -gamma_neg * mu   (negative edge threshold)
    """

    def threshold(
        self, diff: np.ndarray, gamma: float, gamma_neg: float
    ) -> np.ndarray:
        mu = np.mean(np.abs(diff))
        T = gamma * mu
        T_ = -gamma_neg * mu
        edge = np.zeros(len(diff), dtype=np.uint8)
        edge[diff >= T] = 1
        edge[diff <= T_] = 1
        return edge


class PPQECCorrector:
    """Post-Quantum Error Correction — reduces FP/FN across crops (Section 2.8.3).

    Decisions are based on quantum probability P = |diff|^2.
    """

    def correct(
        self,
        edge_crops: List[np.ndarray],
        diff_crops: List[np.ndarray],
    ) -> List[np.ndarray]:
        corrected = [crop.copy() for crop in edge_crops]
        for a in range(1, len(edge_crops)):
            prev = corrected[a - 1]
            curr = corrected[a].copy()
            N = len(curr)
            p_prev = diff_crops[a - 1] ** 2
            p_curr = diff_crops[a] ** 2
            if np.all(curr == 0) and np.any(prev == 1):
                for i in range(N):
                    if prev[i] == 1 and p_prev[i] > p_curr[i]:
                        curr[i] = 1
            elif np.all(curr == 1) and np.any(prev == 0):
                for i in range(N):
                    if prev[i] == 0 and p_prev[i] > p_curr[i]:
                        curr[i] = 0
            corrected[a] = curr
        return corrected


class HQEHEDPipeline:
    """Full HQEHED-AMT pipeline using Qiskit Quantum Simulator (Section 3.3).

    Steps per paper (Rengasamy et al., 2025):
        3.3.1  QPIE   — encode pixels as quantum amplitudes
        3.3.2  H + UD — compute gradient via Hadamard + Unitary Decremental
        3.3.3  AMT    — adaptive mean thresholding
        3.3.4  OR     — combine horizontal + vertical scans
        3.3.5  PPQEC  — post-quantum error correction across crops
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._encoder = QPIEEncoder()
        self._had_edge = HadamardEdgeAmplitude()
        self._amt = AdaptiveMeanThreshold()
        self._ppqec = PPQECCorrector()

    def detect(
        self,
        img: np.ndarray,
        crop_size: Optional[int] = None,
        gamma: Optional[float] = None,
        gamma_neg: Optional[float] = None,
    ) -> np.ndarray:
        """Run the full HQEHED-AMT pipeline.

        Args:
            img       : preprocessed float32 image in [0, 1], shape (H, W)
            crop_size : segment length; will be rounded up to next power of 2
            gamma     : positive edge threshold multiplier
            gamma_neg : negative edge threshold multiplier

        Returns:
            edge_map : uint8 binary array {0, 1}, shape (H, W)
        """
        crop_size = crop_size or self.cfg.crop_size
        gamma = gamma if gamma is not None else self.cfg.gamma
        gamma_neg = gamma_neg if gamma_neg is not None else self.cfg.gamma_neg
        c = _next_pow2(crop_size)
        H, W = img.shape
        edge_h = np.zeros((H, W), dtype=np.uint8)
        edge_v = np.zeros((H, W), dtype=np.uint8)

        # ── Horizontal scan ──────────────────────────────────────
        for row_idx in range(H):
            row = img[row_idx, :]
            row_crops, diff_crops, edge_crops = [], [], []
            for cs in range(0, W, c):
                ce = min(cs + c, W)
                p = self._encoder.encode(row[cs:ce])
                diff = self._had_edge.compute(p)
                diff_crops.append(diff)
                edge_crops.append(self._amt.threshold(diff, gamma, gamma_neg))
                row_crops.append((cs, ce))
            edge_crops = self._ppqec.correct(edge_crops, diff_crops)
            for idx, (cs, ce) in enumerate(row_crops):
                edge_h[row_idx, cs:ce] = edge_crops[idx]

        # ── Vertical scan ────────────────────────────────────────
        for col_idx in range(W):
            col = img[:, col_idx]
            col_crops, diff_crops, edge_crops = [], [], []
            for rs in range(0, H, c):
                re = min(rs + c, H)
                p = self._encoder.encode(col[rs:re])
                diff = self._had_edge.compute(p)
                diff_crops.append(diff)
                edge_crops.append(self._amt.threshold(diff, gamma, gamma_neg))
                col_crops.append((rs, re))
            edge_crops = self._ppqec.correct(edge_crops, diff_crops)
            for idx, (rs, re) in enumerate(col_crops):
                edge_v[rs:re, col_idx] = edge_crops[idx]

        return np.logical_or(edge_h, edge_v).astype(np.uint8)
