"""
features.py — Pixel-wise feature extraction for SVM segmentation.

Feature maps (Section 3.4):
    9 base maps × 3 statistics (center, local_mean, local_std) = 27-D full vector
    6 base maps × 3 statistics                                  = 18-D no-edge variant

Classes:
    FullFeatureExtractor    — 27-D feature vectors (includes HQEHED edge maps)
    NoEdgeFeatureExtractor  — 18-D ablation variant (image features only)
"""

from __future__ import annotations
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..core.config import Config

# ── Constants ─────────────────────────────────────────────────────────────────

_KSIZE = 7   # Local statistics kernel size (must be odd)

# Full 27-D feature map keys (9 maps)
_FM_KEYS_FULL = [
    "intensity",
    "mean",
    "std",
    "grad_mag",
    "grad_cos",
    "grad_sin",
    "edge_flag",
    "edge_density",
    "dist_to_edge",
]

# 18-D ablation keys (6 maps — no edge-derived features)
_FM_KEYS_NOEDGE = [
    "intensity",
    "mean",
    "std",
    "grad_mag",
    "grad_cos",
    "grad_sin",
]

# Type alias: each entry is (center_arr, local_mean_arr, local_std_arr)
FeatureMapDict = Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _local_stats(fm: np.ndarray, k: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute local mean and std of a feature map using a box filter."""
    fm_mean = cv2.blur(fm, k)
    fm_sq_mean = cv2.blur(fm ** 2, k)
    fm_std = np.sqrt(np.clip(fm_sq_mean - fm_mean ** 2, 0.0, None))
    return fm_mean, fm_std


def _build_base_maps(img: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute the 6 image-derived base feature maps."""
    img_f = img.astype(np.float32)
    if img_f.max() > 1.0:
        img_f /= 255.0

    k = (_KSIZE, _KSIZE)
    mean_i = cv2.blur(img_f, k)
    std_i = np.sqrt(np.clip(cv2.blur(img_f ** 2, k) - mean_i ** 2, 0.0, None))

    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    angle = np.arctan2(gy, gx)

    return {
        "intensity": img_f,
        "mean":      mean_i,
        "std":       std_i,
        "grad_mag":  np.sqrt(gx ** 2 + gy ** 2),
        "grad_cos":  np.cos(angle),
        "grad_sin":  np.sin(angle),
    }


def _build_edge_maps(edge_map: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute the 3 edge-derived base feature maps."""
    edge_f = (edge_map > 0).astype(np.float32)
    k = (_KSIZE, _KSIZE)
    edge_den = cv2.blur(edge_f, k)
    inv_u8 = ((1.0 - edge_f) * 255).astype(np.uint8)
    dist = cv2.distanceTransform(inv_u8, cv2.DIST_L2, 5)
    dist_norm = dist / (dist.max() + 1e-8)
    return {
        "edge_flag":    edge_f,
        "edge_density": edge_den,
        "dist_to_edge": dist_norm,
    }


def _build_feature_map_dict(
    img: np.ndarray,
    edge_map: Optional[np.ndarray],
    keys: List[str],
) -> FeatureMapDict:
    """Build a FeatureMapDict for the given key list."""
    k = (_KSIZE, _KSIZE)
    base = _build_base_maps(img)
    if edge_map is not None:
        base.update(_build_edge_maps(edge_map))

    result: FeatureMapDict = {}
    for name in keys:
        fm = base[name]
        fm_mean, fm_std = _local_stats(fm, k)
        result[name] = (fm, fm_mean, fm_std)
    return result


def _feat_vec(fm_dict: FeatureMapDict, y: int, x: int, keys: List[str]) -> List[float]:
    """Extract a feature vector at pixel (y, x) from a FeatureMapDict."""
    feat: List[float] = []
    for name in keys:
        c, m, s = fm_dict[name]
        feat.extend([float(c[y, x]), float(m[y, x]), float(s[y, x])])
    return feat


# ── Public classes ────────────────────────────────────────────────────────────

class FullFeatureExtractor:
    """27-D feature extractor using all 9 feature maps (Section 3.4).

    Feature maps:
        Image-derived (6): intensity, mean, std, grad_mag, grad_cos, grad_sin
        Edge-derived  (3): edge_flag, edge_density, dist_to_edge
        Per map: [center, local_mean(7×7), local_std(7×7)] → 3 stats
        Total: 9 × 3 = 27 features per pixel
    """

    n_features: int = 27
    _keys = _FM_KEYS_FULL

    def build_feature_maps(
        self, img: np.ndarray, edge_map: np.ndarray
    ) -> FeatureMapDict:
        """Pre-compute all feature maps for an image.

        Call once per image; reuse the result for both sampling and
        full-image prediction to guarantee identical features.
        """
        return _build_feature_map_dict(img, edge_map, self._keys)

    def feature_vector(
        self, fm_dict: FeatureMapDict, y: int, x: int
    ) -> List[float]:
        """Extract a 27-D feature vector at pixel (y, x)."""
        return _feat_vec(fm_dict, y, x, self._keys)

    def sample_features(
        self,
        img: np.ndarray,
        edge_map: np.ndarray,
        mask: np.ndarray,
        n_samples: int,
        cfg: Config,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample pixel features with 1:3 tumor:non-tumor ratio.

        Args:
            img       : preprocessed float32 image in [0, 1]
            edge_map  : binary edge map {0, 1}
            mask      : ground-truth binary mask {0, 1}
            n_samples : approximate total samples (split 1:3)
            cfg       : Config (unused here, kept for API consistency)

        Returns:
            X : float32 array, shape (N, 27)
            y : int32 label array, shape (N,)
        """
        fm = self.build_feature_maps(img, edge_map)
        tumor_coords = list(zip(*np.where(mask == 1)))
        nontumor_coords = list(zip(*np.where(mask == 0)))

        n_tumor = min(n_samples // 4, len(tumor_coords))
        n_nontumor = min(n_tumor * 3, len(nontumor_coords))
        if n_tumor == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

        coords = (
            random.sample(tumor_coords, n_tumor)
            + random.sample(nontumor_coords, n_nontumor)
        )
        labels = [1] * n_tumor + [0] * n_nontumor
        features = [self.feature_vector(fm, y, x) for (y, x) in coords]
        return (
            np.array(features, dtype=np.float32),
            np.array(labels, dtype=np.int32),
        )

    def predict_full_image(
        self,
        img: np.ndarray,
        edge_map: np.ndarray,
        clf,
        scaler,
    ) -> np.ndarray:
        """Predict segmentation mask for the full image (vectorized).

        Args:
            img      : preprocessed float32 image in [0, 1]
            edge_map : binary edge map {0, 1}
            clf      : fitted sklearn classifier
            scaler   : fitted StandardScaler

        Returns:
            pred_mask : uint8 array {0, 1}, shape (H, W)
        """
        H, W = img.shape
        fm = self.build_feature_maps(img, edge_map)
        feat_arrays = []
        for name in self._keys:
            c, m, s = fm[name]
            feat_arrays.extend([c.ravel(), m.ravel(), s.ravel()])
        X_full = np.column_stack(feat_arrays).astype(np.float32)
        preds = clf.predict(scaler.transform(X_full))
        return preds.reshape(H, W).astype(np.uint8)


class NoEdgeFeatureExtractor(FullFeatureExtractor):
    """18-D ablation variant — image features only (no edge maps).

    Excludes: edge_flag, edge_density, dist_to_edge.
    Used to quantify the contribution of HQEHED edge features.
    """

    n_features: int = 18
    _keys = _FM_KEYS_NOEDGE

    def build_feature_maps(
        self, img: np.ndarray, edge_map: Optional[np.ndarray] = None
    ) -> FeatureMapDict:
        """Build 18-D feature maps (edge_map is ignored)."""
        return _build_feature_map_dict(img, None, self._keys)

    def predict_full_image(
        self,
        img: np.ndarray,
        edge_map: Optional[np.ndarray],
        clf,
        scaler,
    ) -> np.ndarray:
        """Predict segmentation mask using 18-D features (edge_map ignored)."""
        H, W = img.shape
        fm = self.build_feature_maps(img, None)
        feat_arrays = []
        for name in self._keys:
            c, m, s = fm[name]
            feat_arrays.extend([c.ravel(), m.ravel(), s.ravel()])
        X_full = np.column_stack(feat_arrays).astype(np.float32)
        preds = clf.predict(scaler.transform(X_full))
        return preds.reshape(H, W).astype(np.uint8)
