"""
test_metrics.py — Unit tests for EdgeMetrics, SegmentationMetrics, ImageQualityMetrics.
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.config import Config
from src.evaluation.metrics import EdgeMetrics, SegmentationMetrics, ImageQualityMetrics


@pytest.fixture
def cfg():
    return Config(output_dir="./results_test")


@pytest.fixture
def edge_metrics(cfg):
    return EdgeMetrics(cfg)


@pytest.fixture
def seg_metrics():
    return SegmentationMetrics()


@pytest.fixture
def iqm():
    return ImageQualityMetrics()


# ── EdgeMetrics ───────────────────────────────────────────────────────────────

class TestFigureOfMerit:
    def test_perfect_detection(self, edge_metrics):
        """FOM = 1.0 when detected edges exactly match ground truth."""
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[5, 5] = 1
        fom = edge_metrics.figure_of_merit(gt.copy(), gt.copy())
        assert abs(fom - 1.0) < 1e-6

    def test_both_empty(self, edge_metrics):
        """FOM = 1.0 when both maps are empty (no edges)."""
        empty = np.zeros((10, 10), dtype=np.uint8)
        fom = edge_metrics.figure_of_merit(empty, empty)
        assert fom == 1.0

    def test_no_detection(self, edge_metrics):
        """FOM = 0.0 when nothing is detected but GT has edges."""
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[5, 5] = 1
        det = np.zeros_like(gt)
        fom = edge_metrics.figure_of_merit(det, gt)
        assert fom == 0.0

    def test_no_gt(self, edge_metrics):
        """FOM = 0.0 when GT is empty but detection has edges."""
        det = np.zeros((10, 10), dtype=np.uint8)
        det[5, 5] = 1
        gt = np.zeros_like(det)
        fom = edge_metrics.figure_of_merit(det, gt)
        assert fom == 0.0

    def test_fom_range(self, edge_metrics):
        """FOM must be in [0, 1]."""
        rng = np.random.default_rng(42)
        det = (rng.random((20, 20)) > 0.8).astype(np.uint8)
        gt  = (rng.random((20, 20)) > 0.8).astype(np.uint8)
        fom = edge_metrics.figure_of_merit(det, gt)
        assert 0.0 <= fom <= 1.0


class TestEdgeMetricsCompute:
    def test_precision_recall_formula(self, edge_metrics):
        """Verify Precision = TP/(TP+FP) and Recall = TP/(TP+FN)."""
        det = np.array([[1, 1, 0, 0]], dtype=np.uint8)
        gt  = np.array([[1, 0, 1, 0]], dtype=np.uint8)
        r = edge_metrics.compute(det, gt)
        # TP=1, FP=1, FN=1
        assert abs(r["Precision"] - 0.5) < 1e-4
        assert abs(r["Recall"]    - 0.5) < 1e-4

    def test_f1_is_harmonic_mean(self, edge_metrics):
        """F1 = 2*P*R / (P+R)."""
        det = np.array([[1, 1, 0]], dtype=np.uint8)
        gt  = np.array([[1, 0, 1]], dtype=np.uint8)
        r = edge_metrics.compute(det, gt)
        expected_f1 = 2 * r["Precision"] * r["Recall"] / (r["Precision"] + r["Recall"] + 1e-8)
        assert abs(r["F1"] - expected_f1) < 1e-4

    def test_counts_are_integers(self, edge_metrics):
        det = np.array([[1, 0, 1]], dtype=np.uint8)
        gt  = np.array([[1, 1, 0]], dtype=np.uint8)
        r = edge_metrics.compute(det, gt)
        assert isinstance(r["TP"], int)
        assert isinstance(r["FP"], int)
        assert isinstance(r["FN"], int)


# ── SegmentationMetrics ───────────────────────────────────────────────────────

class TestSegmentationMetrics:
    def test_dice_perfect(self, seg_metrics):
        mask = np.ones((5, 5), dtype=np.uint8)
        assert abs(seg_metrics.dice(mask, mask) - 1.0) < 1e-6

    def test_dice_no_overlap(self, seg_metrics):
        pred = np.zeros((4, 4), dtype=np.uint8)
        pred[:2, :] = 1
        gt = np.zeros((4, 4), dtype=np.uint8)
        gt[2:, :] = 1
        assert seg_metrics.dice(pred, gt) < 1e-4

    def test_iou_perfect(self, seg_metrics):
        mask = np.ones((5, 5), dtype=np.uint8)
        assert abs(seg_metrics.iou(mask, mask) - 1.0) < 1e-6

    def test_iou_no_overlap(self, seg_metrics):
        pred = np.zeros((4, 4), dtype=np.uint8)
        pred[:2, :] = 1
        gt = np.zeros((4, 4), dtype=np.uint8)
        gt[2:, :] = 1
        assert seg_metrics.iou(pred, gt) < 1e-4

    def test_compute_keys(self, seg_metrics):
        pred = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        gt   = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        r = seg_metrics.compute(pred, gt)
        for key in ["Accuracy", "Sensitivity", "Specificity", "Dice", "IoU"]:
            assert key in r

    def test_compute_range(self, seg_metrics):
        rng = np.random.default_rng(0)
        pred = (rng.random((20, 20)) > 0.5).astype(np.uint8)
        gt   = (rng.random((20, 20)) > 0.5).astype(np.uint8)
        r = seg_metrics.compute(pred, gt)
        for v in r.values():
            assert 0.0 <= v <= 1.0 + 1e-6

    def test_from_counts(self, seg_metrics):
        r = seg_metrics.from_counts(tp=10, tn=80, fp=5, fn=5)
        assert abs(r["Accuracy"] - 0.9) < 1e-4
        assert abs(r["Sensitivity"] - 10 / 15) < 1e-4
        assert abs(r["Specificity"] - 80 / 85) < 1e-4


# ── ImageQualityMetrics ───────────────────────────────────────────────────────

class TestImageQualityMetrics:
    def test_mse_zero_identical(self, iqm):
        img = np.ones((10, 10), dtype=np.float64) * 0.5
        mse, psnr = iqm.mse_psnr(img, img)
        assert mse == 0.0
        assert psnr == 100.0

    def test_psnr_formula(self, iqm):
        """PSNR = 10 * log10(R^2 / MSE) with R=255."""
        src = np.zeros((4, 4), dtype=np.float64)
        edg = np.ones((4, 4), dtype=np.float64)
        mse, psnr = iqm.mse_psnr(src, edg)
        expected_psnr = 10.0 * np.log10(255.0 ** 2 / mse)
        assert abs(psnr - expected_psnr) < 1e-6

    def test_mse_positive(self, iqm):
        src = np.zeros((5, 5), dtype=np.float64)
        edg = np.ones((5, 5), dtype=np.float64)
        mse, _ = iqm.mse_psnr(src, edg)
        assert mse > 0.0

    def test_psnr_decreases_with_noise(self, iqm):
        """Higher noise → lower PSNR."""
        src = np.zeros((10, 10), dtype=np.float64)
        low_noise  = src + 0.01
        high_noise = src + 0.5
        _, psnr_low  = iqm.mse_psnr(src, low_noise)
        _, psnr_high = iqm.mse_psnr(src, high_noise)
        assert psnr_low > psnr_high
