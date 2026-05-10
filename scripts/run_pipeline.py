"""
run_pipeline.py — End-to-end HQEHED-AMT + PSO-SVM pipeline runner.

Usage:
    python scripts/run_pipeline.py --data_root ../dataset --output_dir ./results
    python scripts/run_pipeline.py --config configs/default.yaml
"""

from __future__ import annotations
import argparse
import random
import sys
import os

import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Make src importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.config import Config
from src.core.dataset import DatasetLoader, Preprocessor
from src.core.quantum_edge import HQEHEDPipeline
from src.core.classic_edge import CannyDetector
from src.models.features import FullFeatureExtractor
from src.models.pso_optimizer import PSOOptimizer
from src.models.segmentation import SVMSegmenter
from src.evaluation.metrics import EdgeMetrics, SegmentationMetrics, ImageQualityMetrics
from src.evaluation.grid_search import GridSearchTuner
from src.visualization.visualizer import (
    PanelVisualizer, ErrorMapVisualizer,
    PSOConvergenceVisualizer, MetricsBarVisualizer,
)


def parse_args():
    p = argparse.ArgumentParser(description="HQEHED-AMT + PSO-SVM Brain Tumor Segmentation")
    p.add_argument("--data_root",  default="../dataset")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--max_images", type=int, default=30)
    p.add_argument("--gamma",      type=float, default=None, help="Override gamma")
    p.add_argument("--crop_size",  type=int,   default=None, help="Override crop_size")
    p.add_argument("--skip_grid",  action="store_true", help="Skip grid search")
    p.add_argument("--skip_pso",   action="store_true", help="Skip PSO (use defaults)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_images=args.max_images,
    )
    if args.gamma:
        cfg.gamma = args.gamma
    if args.crop_size:
        cfg.crop_size = args.crop_size

    np.random.seed(42); random.seed(42)

    # ── 1. Load dataset ───────────────────────────────────────────
    loader   = DatasetLoader(cfg)
    prep     = Preprocessor(cfg)
    pipeline = HQEHEDPipeline(cfg)
    canny    = CannyDetector()

    pairs = loader.load_pairs()
    if not pairs:
        print(f"[ERROR] No data found at {cfg.data_root}")
        sys.exit(1)
    pairs = pairs[:cfg.max_images]
    print(f"Dataset: {len(pairs)} image pairs loaded.")

    # ── 2. Grid search (optional) ─────────────────────────────────
    best_gamma    = cfg.gamma
    best_gamma_neg = cfg.gamma_neg
    best_crop     = cfg.crop_size

    if not args.skip_grid:
        tuner = GridSearchTuner(cfg)
        gs_results = tuner.search(
            pairs,
            search_grid={"gamma": [0.10, 0.20, 0.30, 0.50, 0.70, 0.95],
                         "crop_size": [16, 32, 64, 128]},
            fixed={"gamma_neg": 0.95},
            max_images=10,
        )
        tuner.print_top(gs_results)
        best = gs_results[0]
        best_gamma    = best.get("gamma",     best_gamma)
        best_crop     = best.get("crop_size", best_crop)
        print(f"\nBest params: gamma={best_gamma}  crop_size={best_crop}")

    # ── 3. Quick visualization (5 samples) ────────────────────────
    quick_samples = pairs[:5]
    iqm  = ImageQualityMetrics()
    em   = EdgeMetrics(cfg)
    panel_viz  = PanelVisualizer(cfg)
    error_viz  = ErrorMapVisualizer(cfg)
    results_quick = []

    print("\nRunning quick visualization ...")
    for img_path, mask_path in tqdm(quick_samples, desc="Quick-Viz"):
        img_raw, gt_mask = loader.load_image_mask(img_path, mask_path)
        img_pre          = prep.process(img_raw)
        edge_hqehed      = pipeline.detect(img_pre, best_crop, best_gamma, best_gamma_neg)
        edge_canny       = canny.detect(img_pre)
        gt_edge          = cv2.Canny((gt_mask * 255).astype(np.uint8), 50, 150) // 255

        mse_h, psnr_h = iqm.mse_psnr(img_pre, edge_hqehed.astype(np.float64))
        mse_c, psnr_c = iqm.mse_psnr(img_pre, edge_canny.astype(np.float64))
        results_quick.append({
            "img_raw": img_raw, "img_pre": img_pre, "gt_mask": gt_mask,
            "gt_edge": gt_edge, "edge_hqehed_raw": edge_hqehed,
            "edge_hqehed": edge_hqehed, "edge_canny": edge_canny,
            "psnr_hqehed_raw": psnr_h, "mse_hqehed_raw": mse_h,
            "psnr_hqehed": psnr_h, "mse_hqehed": mse_h,
            "psnr_canny": psnr_c, "mse_canny": mse_c,
            "fom_hqehed_raw": em.figure_of_merit(edge_hqehed, gt_edge),
            "fom_canny": em.figure_of_merit(edge_canny, gt_edge),
            "gamma": best_gamma, "crop_size": best_crop,
        })

    for i, r in enumerate(results_quick, 1):
        panel_viz.plot(r, i)
        error_viz.plot(r, i)

    # ── 4. PSO hyperparameter search ─────────────────────────────
    extractor = FullFeatureExtractor()
    C_opt, g_opt = cfg.pso_c_range[0], cfg.pso_g_range[0]

    if not args.skip_pso:
        pso_subset = min(10, len(pairs))
        pso_X, pso_y = [], []
        print("\nCollecting PSO training data ...")
        for img_path, mask_path in tqdm(pairs[:pso_subset], desc="PSO data"):
            img_raw, mask = loader.load_image_mask(img_path, mask_path)
            img_pre = prep.process(img_raw)
            edge_map = pipeline.detect(img_pre, best_crop, best_gamma, best_gamma_neg)
            X, y = extractor.sample_features(img_pre, edge_map, mask, 300, cfg)
            if len(X) > 0:
                pso_X.append(X); pso_y.append(y)

        X_pso = np.vstack(pso_X); y_pso = np.concatenate(pso_y)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_pso, y_pso, test_size=0.3, random_state=42, stratify=y_pso
        )
        sc_pso = StandardScaler()
        X_tr_s = sc_pso.fit_transform(X_tr)
        X_val_s = sc_pso.transform(X_val)

        opt = PSOOptimizer(X_tr_s, y_tr, X_val_s, y_val, cfg)
        C_opt, g_opt, pso_history = opt.optimize(verbose=True)
        print(f"\n  C* = {C_opt:.6f}  |  g* = {g_opt:.6f}")
        PSOConvergenceVisualizer(cfg).plot(pso_history)

    # ── 5. Per-image SVM evaluation ───────────────────────────────
    seg_metrics = SegmentationMetrics()
    per_image_results = []
    print(f"\nPer-image evaluation ({len(pairs)} images) ...")
    for img_path, mask_path in tqdm(pairs, desc="Per-image"):
        img_raw, gt_mask = loader.load_image_mask(img_path, mask_path)
        img_pre  = prep.process(img_raw)
        edge_map = pipeline.detect(img_pre, best_crop, best_gamma, best_gamma_neg)
        segmenter = SVMSegmenter(cfg, C=C_opt, gamma=g_opt)
        try:
            _, y_te, y_pred, _ = segmenter.fit(img_pre, edge_map, gt_mask)
            tp = int(np.sum((y_pred==1)&(y_te==1))); tn = int(np.sum((y_pred==0)&(y_te==0)))
            fp = int(np.sum((y_pred==1)&(y_te==0))); fn = int(np.sum((y_pred==0)&(y_te==1)))
            eps = 1e-8
            gt_edge_ = cv2.Canny((gt_mask*255).astype(np.uint8), 50, 150) // 255
            per_image_results.append({
                "metrics": {
                    "Accuracy":    (tp+tn)/(tp+tn+fp+fn+eps),
                    "Sensitivity": tp/(tp+fn+eps),
                    "Specificity": tn/(tn+fp+eps),
                    "Dice":        2*tp/(2*tp+fp+fn+eps),
                    "IoU":         tp/(tp+fp+fn+eps),
                },
                "fom": em.figure_of_merit(edge_map, gt_edge_),
            })
        except ValueError:
            continue

    if per_image_results:
        avg = {k: float(np.mean([r["metrics"][k] for r in per_image_results]))
               for k in per_image_results[0]["metrics"]}
        avg_fom = float(np.mean([r["fom"] for r in per_image_results]))
        print(f"\nAverage metrics ({len(per_image_results)} images):")
        for k, v in avg.items():
            print(f"    {k:<15} {v:.4f}")
        print(f"    {'FOM':<15} {avg_fom:.4f}")
        MetricsBarVisualizer(cfg).plot(avg, avg_fom)

    print(f"\nDone. Results saved to: {cfg.output_dir}/")


if __name__ == "__main__":
    main()
