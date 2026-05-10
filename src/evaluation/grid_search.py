"""
grid_search.py — Grid search tuner for HQEHED-AMT parameters.

Exhaustively evaluates combinations of gamma and crop_size on a small
dataset subset, scoring each by mean PSNR (higher = better).

Class:
    GridSearchTuner — run search, print top-N results
"""

from __future__ import annotations
import itertools
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from ..core.config import Config
from ..core.dataset import DatasetLoader, Preprocessor
from ..core.quantum_edge import HQEHEDPipeline
from .metrics import ImageQualityMetrics


class GridSearchTuner:
    """Exhaustive grid search over HQEHED-AMT parameters.

    Scoring criterion: mean PSNR between source image and edge map
    (Eq. 2.18–2.19, Section 3.4 of the paper).

    Args:
        cfg : Config instance (provides defaults and output_dir)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._loader = DatasetLoader(cfg)
        self._prep = Preprocessor(cfg)
        self._pipeline = HQEHEDPipeline(cfg)
        self._iqm = ImageQualityMetrics()

    def search(
        self,
        pairs: List,
        search_grid: Dict[str, List[Any]],
        fixed: Optional[Dict[str, Any]] = None,
        max_images: int = 10,
    ) -> List[Dict[str, Any]]:
        """Run grid search and return results sorted by PSNR (descending).

        Args:
            pairs       : list of (image_path, mask_path) tuples
            search_grid : dict mapping parameter name → list of values.
                          Supported keys: ``"gamma"``, ``"crop_size"``.
            fixed       : dict of fixed parameters (e.g. ``{"gamma_neg": 0.95}``).
                          Defaults to cfg values if not provided.
            max_images  : number of images to evaluate per combination

        Returns:
            List of result dicts, each containing the parameter values plus
            ``"mean_mse"``, ``"mean_psnr"``, and ``"score"`` (= mean_psnr).
        """
        fixed = fixed or {}
        gamma_neg = fixed.get("gamma_neg", self.cfg.gamma_neg)
        gauss_ksize = fixed.get("gauss_ksize", self.cfg.gauss_ksize)
        gauss_sigma = fixed.get("gauss_sigma", self.cfg.gauss_sigma)

        subset = pairs[:max_images]
        param_keys = list(search_grid.keys())
        combinations = list(itertools.product(*[search_grid[k] for k in param_keys]))

        print(
            f"Grid Search: {len(combinations)} combinations × {len(subset)} images"
        )
        results: List[Dict[str, Any]] = []

        for combo in tqdm(combinations, desc="Grid Search"):
            params = dict(zip(param_keys, combo))
            gamma = params.get("gamma", self.cfg.gamma)
            crop_size = params.get("crop_size", self.cfg.crop_size)

            mses, psnrs = [], []
            for img_path, mask_path in subset:
                img_raw, _ = self._loader.load_image_mask(img_path, mask_path)
                img_pre = self._prep.process(img_raw, gauss_ksize, gauss_sigma)
                edge_hq = self._pipeline.detect(img_pre, crop_size, gamma, gamma_neg)

                mse_val, psnr_val = self._iqm.mse_psnr(
                    img_pre.astype(np.float64),
                    edge_hq.astype(np.float64),
                )
                mses.append(mse_val)
                psnrs.append(psnr_val)

            mean_mse = float(np.mean(mses))
            mean_psnr = float(np.mean(psnrs))
            results.append(
                {
                    **params,
                    "mean_mse":  mean_mse,
                    "mean_psnr": mean_psnr,
                    "score":     mean_psnr,   # higher is better
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def print_top(self, results: List[Dict[str, Any]], n: int = 5) -> None:
        """Print the top-N results in a formatted table.

        Args:
            results : sorted list returned by :meth:`search`
            n       : number of top results to display
        """
        if not results:
            print("No grid search results to display.")
            return

        param_keys = [
            k for k in results[0] if k not in ("mean_mse", "mean_psnr", "score")
        ]
        header = (
            "  "
            + "  ".join(f"{k:>10}" for k in param_keys)
            + f"  {'PSNR (dB)':>10}  {'MSE':>10}"
        )
        print(f"\nTop-{n} configurations (PSNR ↑, MSE ↓):")
        print(header)
        print("  " + "─" * (len(header) - 2))
        for r in results[:n]:
            row = (
                "  "
                + "  ".join(f"{r[k]:>10}" for k in param_keys)
                + f"  {r['mean_psnr']:>10.4f}  {r['mean_mse']:>10.6f}"
            )
            print(row)

        best = results[0]
        print(
            f"\nBest: "
            + "  ".join(f"{k}={best[k]}" for k in param_keys)
            + f"  PSNR={best['mean_psnr']:.4f} dB  MSE={best['mean_mse']:.6f}"
        )
        print("  → Copy to Config or pass as --gamma / --crop_size arguments.")
