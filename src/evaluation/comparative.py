"""
comparative.py — Per-class, multi-method comparative evaluator.

Evaluates glioma / meningioma / pituitary separately across multiple
edge detection methods, computing FOM + segmentation metrics per image.

Class:
    ComparativeEvaluator — run evaluation, aggregate results
"""

from __future__ import annotations
import random
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from ..core.config import Config
from ..core.dataset import DatasetLoader, Preprocessor
from ..models.features import FullFeatureExtractor
from .metrics import EdgeMetrics, SegmentationMetrics


class ComparativeEvaluator:
    """Evaluate multiple edge detectors per tumor class.

    For each (class, method, image):
        1. Detect edges with the given method.
        2. Compute FOM against the GT edge map.
        3. Sample pixel features, split 70/30, train SVM(C*, g*).
        4. Compute segmentation metrics on the test split.

    Args:
        cfg   : Config instance
        C_opt : PSO-optimized SVM penalty parameter
        g_opt : PSO-optimized RBF gamma
    """

    def __init__(self, cfg: Config, C_opt: float, g_opt: float):
        self.cfg = cfg
        self.C_opt = C_opt
        self.g_opt = g_opt
        self._loader = DatasetLoader(cfg)
        self._prep = Preprocessor(cfg)
        self._extractor = FullFeatureExtractor()
        self._edge_metrics = EdgeMetrics(cfg)
        self._seg_metrics = SegmentationMetrics()

    # ── Public API ────────────────────────────────────────────────

    def evaluate(
        self,
        class_pairs: Dict[str, List],
        detectors: Dict[str, Callable],
        n_per_class: int = 15,
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Run full comparative evaluation.

        Args:
            class_pairs : dict mapping class_name → list of (img_path, mask_path)
            detectors   : dict mapping method_name → callable(img_pre) → edge_map
            n_per_class : max images to evaluate per class

        Returns:
            Nested dict: results[class_name][method_name] = list of per-image dicts.
            Each per-image dict contains: fom, tp, tn, fp, fn, dice, iou, acc, sen, spe.
        """
        results: Dict[str, Dict[str, List]] = {
            tc: {m: [] for m in detectors} for tc in class_pairs
        }

        for tc, pairs in class_pairs.items():
            subset = pairs[:n_per_class]
            print(f"\nEvaluating class [{tc.upper()}] ({len(subset)} images) ...")
            for img_p, msk_p in tqdm(subset, desc=f"  {tc}"):
                img_raw, gt_mask = self._loader.load_image_mask(img_p, msk_p)
                img_pre = self._prep.process(img_raw)
                for method_name, edge_fn in detectors.items():
                    edge_map = edge_fn(img_pre)
                    r = self._eval_single(img_pre, gt_mask, edge_map)
                    if r is not None:
                        results[tc][method_name].append(r)

        return results

    def aggregate(
        self,
        results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Aggregate per-image results into mean metrics.

        Returns:
            summary[class_name][method_name] = dict of mean metric values
            plus cumulative TP/TN/FP/FN counts.
        """
        summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
        metric_keys = ["fom", "dice", "iou", "acc", "sen", "spe"]
        count_keys = ["tp", "tn", "fp", "fn"]

        for tc, methods in results.items():
            summary[tc] = {}
            for m, rs in methods.items():
                if not rs:
                    continue
                summary[tc][m] = {
                    k: float(np.mean([r[k] for r in rs])) for k in metric_keys
                }
                for ck in count_keys:
                    summary[tc][m][ck.upper()] = sum(r[ck] for r in rs)
        return summary

    def print_summary(
        self,
        summary: Dict[str, Dict[str, Dict[str, Any]]],
        methods: List[str],
    ) -> None:
        """Print a formatted summary table to stdout."""
        seg_keys = ["acc", "sen", "spe", "dice", "iou"]
        seg_labels = {
            "acc": "Accuracy",
            "sen": "Sensitivity",
            "spe": "Specificity",
            "dice": "Dice",
            "iou": "IoU",
        }
        W = 14

        def _row(label, vals):
            return (
                f"  {label:<18}"
                + "".join(
                    f"{v:>{W}.4f}" if isinstance(v, float) else f"{v:>{W}}"
                    for v in vals
                )
            )

        for tc, methods_data in summary.items():
            print(f"\n  ▌ {tc.upper()}")
            header = f"  {'Metric':<18}" + "".join(f"{m:>{W}}" for m in methods)
            print(header)
            print("  " + "─" * (18 + W * len(methods)))
            # Edge metrics
            print("  [Edge Detection]")
            fom_vals = [methods_data.get(m, {}).get("fom", float("nan")) for m in methods]
            print(_row("FOM (Pratt's)", fom_vals))
            # Segmentation metrics
            print("  [Segmentation]")
            for k in seg_keys:
                vals = [methods_data.get(m, {}).get(k, float("nan")) for m in methods]
                print(_row(seg_labels[k], vals))

    # ── Internal ──────────────────────────────────────────────────

    def _eval_single(
        self,
        img_pre: np.ndarray,
        gt_mask: np.ndarray,
        edge_map: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate one (image, edge_map) pair. Returns None if no tumor pixels."""
        # FOM
        gt_edge = cv2.Canny((gt_mask * 255).astype(np.uint8), 50, 150) // 255
        fom_val = self._edge_metrics.figure_of_merit(edge_map, gt_edge)

        # Feature sampling
        fm = self._extractor.build_feature_maps(img_pre, edge_map)
        tumor_c = list(zip(*np.where(gt_mask == 1)))
        nontumor_c = list(zip(*np.where(gt_mask == 0)))
        n_t = min(self.cfg.eval_n_tumor, len(tumor_c))
        n_nt = min(self.cfg.eval_n_nontumor, len(nontumor_c))
        if n_t == 0:
            return None

        sampled = random.sample(tumor_c, n_t) + random.sample(nontumor_c, n_nt)
        labels = [1] * n_t + [0] * n_nt
        X = np.array(
            [self._extractor.feature_vector(fm, y, x) for (y, x) in sampled],
            dtype=np.float32,
        )
        y = np.array(labels, dtype=np.int32)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        sc = StandardScaler()
        svm = SVC(
            C=self.C_opt,
            gamma=self.g_opt,
            kernel=self.cfg.svm_kernel,
            class_weight=self.cfg.svm_class_weight,
        )
        svm.fit(sc.fit_transform(X_tr), y_tr)
        y_pred = svm.predict(sc.transform(X_te))

        tp = int(np.sum((y_pred == 1) & (y_te == 1)))
        tn = int(np.sum((y_pred == 0) & (y_te == 0)))
        fp = int(np.sum((y_pred == 1) & (y_te == 0)))
        fn = int(np.sum((y_pred == 0) & (y_te == 1)))
        eps = 1e-8
        return {
            "fom": fom_val,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "dice": float(2 * tp / (2 * tp + fp + fn + eps)),
            "iou":  float(tp / (tp + fp + fn + eps)),
            "acc":  float((tp + tn) / (tp + tn + fp + fn + eps)),
            "sen":  float(tp / (tp + fn + eps)),
            "spe":  float(tn / (tn + fp + eps)),
        }
