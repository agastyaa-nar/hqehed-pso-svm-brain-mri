"""
metrics.py — Evaluation metrics for edge detection and segmentation.

Classes:
    EdgeMetrics          — FOM (Pratt's), Precision, Recall, F1
    SegmentationMetrics  — Accuracy, Sensitivity, Specificity, Dice, IoU
    ImageQualityMetrics  — MSE and PSNR (Eq. 2.18–2.19)
"""

from __future__ import annotations
from typing import Dict, Tuple

import numpy as np

from ..core.config import Config


class EdgeMetrics:
    """Metrics for evaluating binary edge maps.

    Args:
        cfg : Config (provides alpha_fom for Pratt's FOM)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def figure_of_merit(
        self,
        edge_det: np.ndarray,
        edge_gt: np.ndarray,
        alpha: float | None = None,
    ) -> float:
        """Pratt's Figure of Merit (Eq. 2.39).

        FOM = (1 / max(Nd, Ng)) * sum_i [ 1 / (1 + alpha * d_i^2) ]

        where d_i is the distance from detected edge point i to the
        nearest ground-truth edge point.

        Args:
            edge_det : detected binary edge map {0, 1}
            edge_gt  : ground-truth binary edge map {0, 1}
            alpha    : scaling constant; defaults to cfg.alpha_fom

        Returns:
            FOM in [0, 1]; 1.0 = perfect detection
        """
        alpha = alpha if alpha is not None else self.cfg.alpha_fom
        det_pts = np.argwhere(edge_det == 1)
        gt_pts = np.argwhere(edge_gt == 1)
        Nd, Ng = len(det_pts), len(gt_pts)
        if Nd == 0 and Ng == 0:
            return 1.0
        if Nd == 0 or Ng == 0:
            return 0.0
        denom = max(Nd, Ng)
        total = sum(
            1.0 / (1.0 + alpha * np.sum((gt_pts - pt) ** 2, axis=1).min())
            for pt in det_pts
        )
        return float(total / denom)

    def compute(
        self,
        edge_det: np.ndarray,
        edge_gt: np.ndarray,
    ) -> Dict[str, float]:
        """Compute Precision, Recall, F1, TP, FP, FN for an edge map.

        Args:
            edge_det : detected binary edge map {0, 1}
            edge_gt  : ground-truth binary edge map {0, 1}

        Returns:
            dict with keys: Precision, Recall, F1, TP, FP, FN
        """
        tp = int(np.sum((edge_det == 1) & (edge_gt == 1)))
        fp = int(np.sum((edge_det == 1) & (edge_gt == 0)))
        fn = int(np.sum((edge_det == 0) & (edge_gt == 1)))
        eps = 1e-8
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        return {
            "Precision": float(prec),
            "Recall":    float(rec),
            "F1":        float(f1),
            "TP":        tp,
            "FP":        fp,
            "FN":        fn,
        }


class SegmentationMetrics:
    """Metrics for evaluating binary segmentation masks."""

    @staticmethod
    def dice(pred: np.ndarray, gt: np.ndarray) -> float:
        """Dice coefficient (F1 for segmentation)."""
        tp = int(np.sum((pred == 1) & (gt == 1)))
        fp = int(np.sum((pred == 1) & (gt == 0)))
        fn = int(np.sum((pred == 0) & (gt == 1)))
        return float(2 * tp / (2 * tp + fp + fn + 1e-8))

    @staticmethod
    def iou(pred: np.ndarray, gt: np.ndarray) -> float:
        """Intersection over Union (Jaccard index)."""
        tp = int(np.sum((pred == 1) & (gt == 1)))
        fp = int(np.sum((pred == 1) & (gt == 0)))
        fn = int(np.sum((pred == 0) & (gt == 1)))
        return float(tp / (tp + fp + fn + 1e-8))

    def compute(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
    ) -> Dict[str, float]:
        """Compute all segmentation metrics.

        Args:
            pred : predicted binary mask {0, 1}
            gt   : ground-truth binary mask {0, 1}

        Returns:
            dict with keys: Accuracy, Sensitivity, Specificity, Dice, IoU
        """
        pred_f = pred.ravel()
        gt_f = gt.ravel()
        tp = int(np.sum((pred_f == 1) & (gt_f == 1)))
        tn = int(np.sum((pred_f == 0) & (gt_f == 0)))
        fp = int(np.sum((pred_f == 1) & (gt_f == 0)))
        fn = int(np.sum((pred_f == 0) & (gt_f == 1)))
        eps = 1e-8
        return {
            "Accuracy":    float((tp + tn) / (tp + tn + fp + fn + eps)),
            "Sensitivity": float(tp / (tp + fn + eps)),
            "Specificity": float(tn / (tn + fp + eps)),
            "Dice":        self.dice(pred, gt),
            "IoU":         self.iou(pred, gt),
        }

    def from_counts(
        self,
        tp: int,
        tn: int,
        fp: int,
        fn: int,
    ) -> Dict[str, float]:
        """Compute metrics directly from confusion matrix counts."""
        eps = 1e-8
        return {
            "Accuracy":    float((tp + tn) / (tp + tn + fp + fn + eps)),
            "Sensitivity": float(tp / (tp + fn + eps)),
            "Specificity": float(tn / (tn + fp + eps)),
            "Dice":        float(2 * tp / (2 * tp + fp + fn + eps)),
            "IoU":         float(tp / (tp + fp + fn + eps)),
        }


class ImageQualityMetrics:
    """Image quality metrics: MSE and PSNR (Eq. 2.18–2.19).

    PSNR uses R = 255 (8-bit dynamic range) as per the paper,
    even when images are normalized to [0, 1].
    """

    @staticmethod
    def mse_psnr(
        src: np.ndarray,
        edg: np.ndarray,
        R: float = 255.0,
    ) -> Tuple[float, float]:
        """Compute MSE and PSNR between source image and edge map.

        Args:
            src : source grayscale image, float in [0, 1]
            edg : edge map, float or uint8 in {0, 1}
            R   : dynamic range for PSNR (default 255 per paper)

        Returns:
            (mse, psnr) — PSNR is capped at 100 dB when MSE = 0
        """
        src_f = src.astype(np.float64)
        edg_f = edg.astype(np.float64)
        mse = float(np.mean((src_f - edg_f) ** 2))
        psnr = float(10.0 * np.log10(R ** 2 / mse)) if mse > 0 else 100.0
        return mse, psnr
