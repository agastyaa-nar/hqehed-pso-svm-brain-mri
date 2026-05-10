"""
segmentation.py — SVM-based tumor segmentation with morphological post-processing.

Classes:
    MorphologicalPostProcessor — MORPH_OPEN + connected-component filtering
    SVMSegmenter               — Per-image SVM train + full-image prediction
"""

from __future__ import annotations
import random
from typing import Optional, Tuple

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from ..core.config import Config
from .features import FullFeatureExtractor


class MorphologicalPostProcessor:
    """Clean a raw segmentation mask using morphological opening and
    connected-component filtering.

    Steps:
        1. Morphological opening (ellipse kernel 5×5, 2 iterations)
           to remove small isolated noise blobs.
        2. Keep at most ``cfg.cc_max_blobs`` connected components
           with area >= ``cfg.cc_min_size`` pixels.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def process(self, pred_raw: np.ndarray) -> np.ndarray:
        """Apply morphological post-processing.

        Args:
            pred_raw : uint8 binary mask {0, 1}, shape (H, W)

        Returns:
            pred_clean : uint8 binary mask {0, 1}, shape (H, W)
        """
        if pred_raw.max() == 0:
            return pred_raw.copy()

        opened = cv2.morphologyEx(
            pred_raw.astype(np.uint8),
            cv2.MORPH_OPEN,
            self._kernel,
            iterations=2,
        )

        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(
            opened, connectivity=8
        )
        # Sort blobs by area (descending), skip background (label 0)
        blobs = sorted(
            [
                (stats[i, cv2.CC_STAT_AREA], i)
                for i in range(1, n_lbl)
                if stats[i, cv2.CC_STAT_AREA] >= self.cfg.cc_min_size
            ],
            reverse=True,
        )

        pred_clean = np.zeros_like(pred_raw, dtype=np.uint8)
        for _, blob_idx in blobs[: self.cfg.cc_max_blobs]:
            pred_clean[lbl == blob_idx] = 1
        return pred_clean


class SVMSegmenter:
    """Per-image SVM segmenter using PSO-optimized hyperparameters.

    Workflow:
        1. Sample pixel features (1:4 tumor:non-tumor ratio).
        2. Split 70% train / 30% test.
        3. Fit StandardScaler + SVC(C=C_opt, gamma=g_opt, kernel='rbf').
        4. Predict on test split → return metrics.
        5. predict_full() → full-image segmentation mask.

    Args:
        cfg   : Config instance
        C     : SVM penalty parameter (from PSO)
        gamma : RBF kernel gamma (from PSO)
    """

    def __init__(self, cfg: Config, C: float = 1.0, gamma: float = 0.1):
        self.cfg = cfg
        self.C = C
        self.gamma = gamma
        self._extractor = FullFeatureExtractor()
        self._post = MorphologicalPostProcessor(cfg)
        self.scaler: Optional[StandardScaler] = None
        self.clf: Optional[SVC] = None

    def fit(
        self,
        img: np.ndarray,
        edge_map: np.ndarray,
        gt_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        """Sample pixels, train SVM, return test predictions.

        Args:
            img      : preprocessed float32 image in [0, 1]
            edge_map : binary edge map {0, 1}
            gt_mask  : ground-truth binary mask {0, 1}

        Returns:
            (X_test_scaled, y_test, y_pred, scaler)

        Raises:
            ValueError: if no tumor pixels are found in gt_mask.
        """
        fm = self._extractor.build_feature_maps(img, edge_map)
        tumor_c = list(zip(*np.where(gt_mask == 1)))
        nontumor_c = list(zip(*np.where(gt_mask == 0)))

        n_t = min(self.cfg.eval_n_tumor, len(tumor_c))
        n_nt = min(self.cfg.eval_n_nontumor, len(nontumor_c))
        if n_t == 0:
            raise ValueError("No tumor pixels found in ground-truth mask.")

        sampled = (
            random.sample(tumor_c, n_t) + random.sample(nontumor_c, n_nt)
        )
        labels = [1] * n_t + [0] * n_nt
        X = np.array(
            [self._extractor.feature_vector(fm, y, x) for (y, x) in sampled],
            dtype=np.float32,
        )
        y = np.array(labels, dtype=np.int32)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        self.scaler = StandardScaler()
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)

        self.clf = SVC(
            C=self.C,
            gamma=self.gamma,
            kernel=self.cfg.svm_kernel,
            class_weight=self.cfg.svm_class_weight,
        )
        self.clf.fit(X_tr_s, y_tr)
        y_pred = self.clf.predict(X_te_s)
        return X_te_s, y_te, y_pred, self.scaler

    def predict_full(
        self,
        img: np.ndarray,
        edge_map: np.ndarray,
        apply_postprocess: bool = True,
    ) -> np.ndarray:
        """Predict full-image segmentation mask.

        Must call :meth:`fit` first.

        Args:
            img               : preprocessed float32 image in [0, 1]
            edge_map          : binary edge map {0, 1}
            apply_postprocess : if True, apply morphological cleaning

        Returns:
            pred_mask : uint8 binary mask {0, 1}, shape (H, W)
        """
        if self.clf is None or self.scaler is None:
            raise RuntimeError("Call fit() before predict_full().")
        pred_raw = self._extractor.predict_full_image(
            img, edge_map, self.clf, self.scaler
        )
        if apply_postprocess:
            return self._post.process(pred_raw)
        return pred_raw
