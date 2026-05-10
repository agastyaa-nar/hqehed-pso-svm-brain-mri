"""
visualizer.py — Dark-theme matplotlib visualization classes for
HQEHED-AMT + PSO-SVM brain tumor segmentation project.

All classes use the GitHub dark theme:
  background : #0d1117
  axes face  : #161b22
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ..core.config import Config

# ── Theme constants ────────────────────────────────────────────────────────────
_BG      = "#0d1117"
_AX      = "#161b22"
_GRID    = "#21262d"
_SPINE   = "#30363d"
_WHITE   = "white"
_PALETTE: Dict[str, str] = {
    "Canny":      "#58a6ff",
    "Sobel":      "#56d364",
    "Prewitt":    "#e3b341",
    "LoG":        "#bc8cff",
    "HQEHED-AMT": "#f78166",
}
_DEFAULT_COLORS: List[str] = [
    "#58a6ff", "#56d364", "#f78166",
    "#e3b341", "#bc8cff", "#39d353",
]


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _apply_dark_theme(fig: plt.Figure, axes) -> None:
    """Apply GitHub dark theme to a figure and all its axes."""
    fig.patch.set_facecolor(_BG)
    ax_list = axes if hasattr(axes, "__iter__") else [axes]
    # Flatten nested arrays (e.g. from plt.subplots with squeeze=False)
    flat: List[plt.Axes] = []
    for item in ax_list:
        if hasattr(item, "__iter__"):
            flat.extend(list(item))
        else:
            flat.append(item)
    for ax in flat:
        ax.set_facecolor(_AX)
        ax.tick_params(colors=_WHITE, labelsize=8)
        ax.xaxis.label.set_color(_WHITE)
        ax.yaxis.label.set_color(_WHITE)
        ax.title.set_color(_WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor(_SPINE)


def _save(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save figure and close it."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _method_color(method: str, idx: int = 0) -> str:
    return _PALETTE.get(method, _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)])


# ══════════════════════════════════════════════════════════════════════════════
# 1. PanelVisualizer
# ══════════════════════════════════════════════════════════════════════════════

class PanelVisualizer:
    """5-column panel: MRI | GT Mask | GT Edge | Canny | HQEHED-AMT.

    Parameters
    ----------
    cfg : Config
        Project configuration (uses cfg.output_dir).
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def plot(
        self,
        mri: np.ndarray,
        gt_mask: np.ndarray,
        gt_edge: np.ndarray,
        canny: np.ndarray,
        hqehed: np.ndarray,
        title: str = "Edge Detection Comparison",
        filename: str = "panel_comparison.png",
    ) -> str:
        """Render and save the 5-column panel.

        Parameters
        ----------
        mri      : Grayscale or RGB MRI image (H×W or H×W×3).
        gt_mask  : Ground-truth binary mask (H×W).
        gt_edge  : Ground-truth edge map (H×W).
        canny    : Canny edge map (H×W).
        hqehed   : HQEHED-AMT edge map (H×W).
        title    : Figure suptitle.
        filename : Output filename (relative to cfg.output_dir).

        Returns
        -------
        str : Absolute path to the saved file.
        """
        cols   = [mri, gt_mask, gt_edge, canny, hqehed]
        labels = ["MRI", "GT Mask", "GT Edge", "Canny", "HQEHED-AMT"]
        cmaps  = ["gray", "gray", "gray", "gray", "hot"]

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        _apply_dark_theme(fig, axes)

        for ax, img, lbl, cmap in zip(axes, cols, labels, cmaps):
            display = img
            if display.ndim == 2:
                ax.imshow(display, cmap=cmap, vmin=0, vmax=255 if display.max() > 1 else 1)
            else:
                ax.imshow(display)
            ax.set_title(lbl, color=_WHITE, fontsize=10, fontweight="bold")
            ax.axis("off")

        fig.suptitle(title, color=_WHITE, fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()

        out = os.path.join(self.cfg.output_dir, filename)
        _save(fig, out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 2. ErrorMapVisualizer
# ══════════════════════════════════════════════════════════════════════════════

class ErrorMapVisualizer:
    """4-panel TP / FP / FN overlay visualizer.

    Panels: Original MRI | TP overlay | FP overlay | FN overlay.
    """

    # Overlay colours (RGBA)
    _TP_COLOR = np.array([0, 255, 0,   160], dtype=np.uint8)   # green
    _FP_COLOR = np.array([255, 0, 0,   160], dtype=np.uint8)   # red
    _FN_COLOR = np.array([255, 165, 0, 160], dtype=np.uint8)   # orange

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @staticmethod
    def _overlay(base_gray: np.ndarray, mask: np.ndarray, color: np.ndarray) -> np.ndarray:
        """Blend a coloured mask onto a grayscale base image."""
        h, w = base_gray.shape[:2]
        rgb = cv2.cvtColor(
            base_gray if base_gray.ndim == 2 else cv2.cvtColor(base_gray, cv2.COLOR_BGR2GRAY),
            cv2.COLOR_GRAY2RGB,
        )
        overlay = rgb.copy()
        overlay[mask > 0] = color[:3]
        alpha = color[3] / 255.0
        blended = cv2.addWeighted(overlay, alpha, rgb, 1 - alpha, 0)
        return blended

    def plot(
        self,
        mri: np.ndarray,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        title: str = "Segmentation Error Map",
        filename: str = "error_map.png",
    ) -> str:
        """Render and save the 4-panel error map.

        Parameters
        ----------
        mri       : Grayscale MRI image (H×W).
        pred_mask : Binary predicted mask (H×W), values 0/1 or 0/255.
        gt_mask   : Binary ground-truth mask (H×W), values 0/1 or 0/255.
        """
        pred = (pred_mask > 0).astype(np.uint8)
        gt   = (gt_mask   > 0).astype(np.uint8)

        tp = ((pred == 1) & (gt == 1)).astype(np.uint8)
        fp = ((pred == 1) & (gt == 0)).astype(np.uint8)
        fn = ((pred == 0) & (gt == 1)).astype(np.uint8)

        panels = [
            (mri,                                    "Original MRI"),
            (self._overlay(mri, tp, self._TP_COLOR), "True Positive (TP)"),
            (self._overlay(mri, fp, self._FP_COLOR), "False Positive (FP)"),
            (self._overlay(mri, fn, self._FN_COLOR), "False Negative (FN)"),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        _apply_dark_theme(fig, axes)

        for ax, (img, lbl) in zip(axes, panels):
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.set_title(lbl, color=_WHITE, fontsize=9, fontweight="bold")
            ax.axis("off")

        # Legend
        patches = [
            mpatches.Patch(color=np.array(self._TP_COLOR[:3]) / 255, label="TP"),
            mpatches.Patch(color=np.array(self._FP_COLOR[:3]) / 255, label="FP"),
            mpatches.Patch(color=np.array(self._FN_COLOR[:3]) / 255, label="FN"),
        ]
        fig.legend(
            handles=patches, loc="lower center", ncol=3,
            facecolor=_AX, edgecolor=_SPINE, labelcolor=_WHITE,
            fontsize=9, bbox_to_anchor=(0.5, -0.05),
        )

        fig.suptitle(title, color=_WHITE, fontsize=12, fontweight="bold")
        fig.tight_layout()

        out = os.path.join(self.cfg.output_dir, filename)
        _save(fig, out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. PSOConvergenceVisualizer
# ══════════════════════════════════════════════════════════════════════════════

class PSOConvergenceVisualizer:
    """Single-panel PSO convergence curve (best fitness vs. iteration)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def plot(
        self,
        history: List[float],
        title: str = "PSO Convergence",
        filename: str = "pso_convergence.png",
        ylabel: str = "Best Fitness (Validation Accuracy)",
    ) -> str:
        """Render and save the PSO convergence curve.

        Parameters
        ----------
        history  : List of best-fitness values per iteration.
        title    : Figure title.
        filename : Output filename.
        ylabel   : Y-axis label.
        """
        iterations = np.arange(1, len(history) + 1)

        fig, ax = plt.subplots(figsize=(8, 4))
        _apply_dark_theme(fig, ax)

        ax.plot(
            iterations, history,
            color=_PALETTE["HQEHED-AMT"], linewidth=2.0,
            marker="o", markersize=3, markerfacecolor=_PALETTE["HQEHED-AMT"],
        )
        ax.fill_between(iterations, history, alpha=0.15, color=_PALETTE["HQEHED-AMT"])

        ax.set_xlabel("Iteration", color=_WHITE, fontsize=10)
        ax.set_ylabel(ylabel, color=_WHITE, fontsize=10)
        ax.set_title(title, color=_WHITE, fontsize=12, fontweight="bold")
        ax.grid(True, color=_GRID, linestyle="--", linewidth=0.6, alpha=0.7)
        ax.set_xlim(1, len(history))

        # Annotate best value
        best_iter = int(np.argmax(history)) + 1
        best_val  = max(history)
        ax.annotate(
            f"Best: {best_val:.4f}\n@ iter {best_iter}",
            xy=(best_iter, best_val),
            xytext=(best_iter + max(1, len(history) * 0.05), best_val - 0.02),
            color=_WHITE, fontsize=8,
            arrowprops=dict(arrowstyle="->", color=_WHITE, lw=1.0),
        )

        fig.tight_layout()
        out = os.path.join(self.cfg.output_dir, filename)
        _save(fig, out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 4. MetricsBarVisualizer
# ══════════════════════════════════════════════════════════════════════════════

class MetricsBarVisualizer:
    """Grouped bar chart for multiple metrics across multiple methods/classes."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def plot(
        self,
        metrics: Dict[str, List[float]],
        group_labels: List[str],
        title: str = "Metrics Comparison",
        filename: str = "metrics_bar.png",
        ylabel: str = "Score",
        ylim: Tuple[float, float] = (0.0, 1.05),
    ) -> str:
        """Render and save a grouped bar chart.

        Parameters
        ----------
        metrics      : Dict mapping metric name → list of values (one per group).
        group_labels : Labels for each group on the x-axis.
        title        : Figure title.
        filename     : Output filename.
        ylabel       : Y-axis label.
        ylim         : Y-axis limits.
        """
        metric_names = list(metrics.keys())
        n_metrics    = len(metric_names)
        n_groups     = len(group_labels)

        x      = np.arange(n_groups)
        width  = 0.8 / n_metrics
        offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

        fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.5), 5))
        _apply_dark_theme(fig, ax)

        for i, (name, vals) in enumerate(metrics.items()):
            color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
            bars  = ax.bar(x + offsets[i], vals, width=width * 0.9,
                           color=color, label=name, alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    color=_WHITE, fontsize=6.5,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, color=_WHITE, fontsize=9)
        ax.set_ylabel(ylabel, color=_WHITE, fontsize=10)
        ax.set_title(title, color=_WHITE, fontsize=12, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y", color=_GRID, linestyle="--", linewidth=0.6, alpha=0.7)
        ax.legend(
            facecolor=_AX, edgecolor=_SPINE, labelcolor=_WHITE, fontsize=8,
        )

        fig.tight_layout()
        out = os.path.join(self.cfg.output_dir, filename)
        _save(fig, out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 5. PSOAnalysisVisualizer
# ══════════════════════════════════════════════════════════════════════════════

class PSOAnalysisVisualizer:
    """3-panel PSO analysis: convergence curve + C/γ heatmap + SVM landscape.

    Panel layout
    ------------
    [0] PSO convergence curve (best fitness vs. iteration)
    [1] Grid-search accuracy heatmap over log10(C) × log10(γ)
    [2] SVM decision landscape on the PSO-optimised (C, γ) pair
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def plot(
        self,
        history: List[float],
        optimizer: Any,
        gs_heat: np.ndarray,
        C_log: np.ndarray,
        g_log: np.ndarray,
        pso_history: List[float],
        X_pso_tr: np.ndarray,
        y_pso_tr: np.ndarray,
        X_pso_val: np.ndarray,
        y_pso_val: np.ndarray,
        C_opt: float,
        g_opt: float,
        cfg: Optional[Config] = None,
        title: str = "PSO Hyperparameter Optimisation Analysis",
        filename: str = "pso_analysis.png",
    ) -> str:
        """Render and save the 3-panel PSO analysis figure.

        Parameters
        ----------
        history     : Best-fitness list from the main PSO run.
        optimizer   : PSOOptimizer instance (used for metadata annotation).
        gs_heat     : 2-D accuracy array (len(C_log) × len(g_log)) from grid search.
        C_log       : 1-D array of log10(C) values (x-axis of heatmap).
        g_log       : 1-D array of log10(γ) values (y-axis of heatmap).
        pso_history : Best-fitness list (may equal *history*; kept separate for
                      flexibility).
        X_pso_tr    : Training feature matrix used for PSO fitness evaluation.
        y_pso_tr    : Training labels.
        X_pso_val   : Validation feature matrix.
        y_pso_val   : Validation labels.
        C_opt       : Optimal C found by PSO (linear scale).
        g_opt       : Optimal γ found by PSO (linear scale).
        cfg         : Optional override Config (falls back to self.cfg).
        title       : Figure suptitle.
        filename    : Output filename.
        """
        cfg = cfg or self.cfg

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        _apply_dark_theme(fig, axes)

        # ── Panel 0: Convergence ──────────────────────────────────────────────
        ax0 = axes[0]
        iters = np.arange(1, len(pso_history) + 1)
        ax0.plot(iters, pso_history, color=_PALETTE["HQEHED-AMT"], linewidth=2.0,
                 marker="o", markersize=3)
        ax0.fill_between(iters, pso_history, alpha=0.15, color=_PALETTE["HQEHED-AMT"])
        ax0.set_xlabel("Iteration", color=_WHITE, fontsize=9)
        ax0.set_ylabel("Best Validation Accuracy", color=_WHITE, fontsize=9)
        ax0.set_title("PSO Convergence", color=_WHITE, fontsize=10, fontweight="bold")
        ax0.grid(True, color=_GRID, linestyle="--", linewidth=0.6, alpha=0.7)
        best_val = max(pso_history)
        best_it  = int(np.argmax(pso_history)) + 1
        ax0.annotate(
            f"Best={best_val:.4f}\n@iter {best_it}",
            xy=(best_it, best_val),
            xytext=(best_it + max(1, len(pso_history) * 0.08), best_val - 0.025),
            color=_WHITE, fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color=_WHITE, lw=0.8),
        )

        # ── Panel 1: Grid-search heatmap ──────────────────────────────────────
        ax1 = axes[1]
        im = ax1.imshow(
            gs_heat, aspect="auto", origin="lower",
            cmap="YlOrRd",
            extent=[C_log.min(), C_log.max(), g_log.min(), g_log.max()],
        )
        cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=_WHITE)
        cbar.outline.set_edgecolor(_SPINE)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_WHITE, fontsize=7)
        cbar.set_label("Accuracy", color=_WHITE, fontsize=8)

        # Mark PSO optimum
        ax1.scatter(
            [np.log10(C_opt)], [np.log10(g_opt)],
            marker="*", s=200, color=_PALETTE["HQEHED-AMT"],
            zorder=5, label=f"PSO opt\nC={C_opt:.3f}, γ={g_opt:.4f}",
        )
        ax1.set_xlabel("log₁₀(C)", color=_WHITE, fontsize=9)
        ax1.set_ylabel("log₁₀(γ)", color=_WHITE, fontsize=9)
        ax1.set_title("C–γ Accuracy Heatmap", color=_WHITE, fontsize=10, fontweight="bold")
        ax1.legend(facecolor=_AX, edgecolor=_SPINE, labelcolor=_WHITE, fontsize=7)

        # ── Panel 2: SVM decision landscape (PCA 2-D projection) ─────────────
        ax2 = axes[2]
        try:
            from sklearn.decomposition import PCA
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_pso_tr)
            X_va_s = scaler.transform(X_pso_val)

            pca = PCA(n_components=2, random_state=42)
            X_tr_2d = pca.fit_transform(X_tr_s)
            X_va_2d = pca.transform(X_va_s)

            # Fit SVM with optimal params in 2-D PCA space
            svm = SVC(C=C_opt, gamma=g_opt, kernel="rbf")
            svm.fit(X_tr_2d, y_pso_tr)

            # Decision boundary mesh
            x_min, x_max = X_tr_2d[:, 0].min() - 0.5, X_tr_2d[:, 0].max() + 0.5
            y_min, y_max = X_tr_2d[:, 1].min() - 0.5, X_tr_2d[:, 1].max() + 0.5
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 200),
                np.linspace(y_min, y_max, 200),
            )
            Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            ax2.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
            ax2.contour(xx, yy, Z, colors=_SPINE, linewidths=0.8)

            # Scatter validation points
            for cls, color, label in [(0, _PALETTE["Canny"], "Non-tumor"),
                                       (1, _PALETTE["HQEHED-AMT"], "Tumor")]:
                mask = y_pso_val == cls
                ax2.scatter(
                    X_va_2d[mask, 0], X_va_2d[mask, 1],
                    c=color, s=10, alpha=0.6, label=label,
                )
            ax2.set_xlabel("PC 1", color=_WHITE, fontsize=9)
            ax2.set_ylabel("PC 2", color=_WHITE, fontsize=9)
            ax2.legend(facecolor=_AX, edgecolor=_SPINE, labelcolor=_WHITE, fontsize=7)
        except Exception as exc:  # noqa: BLE001
            ax2.text(
                0.5, 0.5, f"Landscape unavailable\n({exc})",
                ha="center", va="center", color=_WHITE, fontsize=8,
                transform=ax2.transAxes,
            )

        ax2.set_title("SVM Decision Landscape (PCA 2-D)", color=_WHITE,
                      fontsize=10, fontweight="bold")

        fig.suptitle(title, color=_WHITE, fontsize=13, fontweight="bold")
        fig.tight_layout()

        out = os.path.join(cfg.output_dir, filename)
        _save(fig, out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 6. ComparativeBarVisualizer
# ══════════════════════════════════════════════════════════════════════════════

class ComparativeBarVisualizer:
    """Grouped bar chart: FOM / F1 / Dice / IoU per tumor class per method.

    Saves to: eval_fom_per_kelas.png
    """

    _METRICS = ["fom", "f1", "dice", "iou"]
    _METRIC_LABELS = ["FOM", "F1-Score", "Dice", "IoU"]

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def plot(
        self,
        summary: Dict[str, Any],
        tumor_classes: List[str],
        methods: List[str],
        filename: str = "eval_fom_per_kelas.png",
    ) -> str:
        """Render and save the comparative bar chart.

        Parameters
        ----------
        summary       : Nested dict  summary[method][class][metric] → float.
        tumor_classes : List of tumor class names (e.g. ["glioma", ...]).
        methods       : List of method names (e.g. ["Canny", "HQEHED-AMT"]).
        filename      : Output filename.
        """
        n_classes = len(tumor_classes)
        n_metrics = len(self._METRICS)
        fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5), sharey=False)
        if n_classes == 1:
            axes = [axes]
        _apply_dark_theme(fig, axes)

        for ax, cls in zip(axes, tumor_classes):
            x      = np.arange(n_metrics)
            width  = 0.8 / len(methods)
            offsets = np.linspace(
                -(len(methods) - 1) / 2,
                (len(methods) - 1) / 2,
                len(methods),
            ) * width

            for i, method in enumerate(methods):
                vals = [
                    float(summary.get(method, {}).get(cls, {}).get(m, 0.0))
                    for m in self._METRICS
                ]
                color = _method_color(method, i)
                bars  = ax.bar(x + offsets[i], vals, width=width * 0.9,
                               color=color, label=method, alpha=0.85)
                for bar, val in zip(bars, vals):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom",
                        color=_WHITE, fontsize=6, rotation=90,
                    )

            ax.set_xticks(x)
            ax.set_xticklabels(self._METRIC_LABELS, color=_WHITE, fontsize=9)
            ax.set_title(cls.capitalize(), color=_WHITE, fontsize=11, fontweight="bold")
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Score", color=_WHITE, fontsize=9)
            ax.grid(True, axis="y", color=_GRID, linestyle="--", linewidth=0.6, alpha=0.7)
            ax.legend(facecolor=_AX, edgecolor=_SPINE, labelcolor=_WHITE, fontsize=7)

        fig.suptitle(
            "Comparative Evaluation: FOM / F1 / Dice / IoU per Class",
            color=_WHITE, fontsize=13, fontweight="bold",
        )
        fig.tight_layout()

        out = os.path.join(self.cfg.output_dir, filename)
        _save(fig, out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 7. SegmentationHeatmapVisualizer
# ══════════════════════════════════════════════════════════════════════════════

class SegmentationHeatmapVisualizer:
    """Seaborn heatmap: method (rows) × metric (columns) per tumor class.

    Saves to: eval_segmentasi_per_kelas.png
    """

    _METRICS = ["fom", "f1", "dice", "iou", "precision", "recall"]
    _METRIC_LABELS = ["FOM", "F1", "Dice", "IoU", "Precision", "Recall"]

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def plot(
        self,
        summary: Dict[str, Any],
        tumor_classes: List[str],
        methods: List[str],
        filename: str = "eval_segmentasi_per_kelas.png",
    ) -> str:
        """Render and save the segmentation heatmap.

        Parameters
        ----------
        summary       : Nested dict  summary[method][class][metric] → float.
        tumor_classes : List of tumor class names.
        methods       : List of method names.
        filename      : Output filename.
        """
        try:
            import seaborn as sns
        except ImportError:
            sns = None

        n_classes = len(tumor_classes)
        fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 4))
        if n_classes == 1:
            axes = [axes]
        _apply_dark_theme(fig, axes)

        for ax, cls in zip(axes, tumor_classes):
            data = np.array([
                [float(summary.get(m, {}).get(cls, {}).get(met, 0.0))
                 for met in self._METRICS]
                for m in methods
            ])

            if sns is not None:
                sns.heatmap(
                    data,
                    ax=ax,
                    annot=True, fmt=".3f", annot_kws={"size": 7, "color": _WHITE},
                    xticklabels=self._METRIC_LABELS,
                    yticklabels=methods,
                    cmap="YlOrRd",
                    vmin=0.0, vmax=1.0,
                    linewidths=0.4, linecolor=_SPINE,
                    cbar_kws={"shrink": 0.8},
                )
                ax.collections[0].colorbar.ax.yaxis.set_tick_params(color=_WHITE)
                plt.setp(
                    ax.collections[0].colorbar.ax.yaxis.get_ticklabels(),
                    color=_WHITE, fontsize=7,
                )
            else:
                im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
                ax.set_xticks(range(len(self._METRIC_LABELS)))
                ax.set_xticklabels(self._METRIC_LABELS, fontsize=8)
                ax.set_yticks(range(len(methods)))
                ax.set_yticklabels(methods, fontsize=8)
                for r in range(data.shape[0]):
                    for c in range(data.shape[1]):
                        ax.text(c, r, f"{data[r, c]:.3f}",
                                ha="center", va="center", color=_WHITE, fontsize=7)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_title(cls.capitalize(), color=_WHITE, fontsize=11, fontweight="bold")
            ax.tick_params(colors=_WHITE, labelsize=8)

        fig.suptitle(
            "Segmentation Metrics Heatmap (Method × Metric per Class)",
            color=_WHITE, fontsize=13, fontweight="bold",
        )
        fig.tight_layout()

        out = os.path.join(self.cfg.output_dir, filename)
        _save(fig, out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 8. ConfusionMatrixVisualizer
# ══════════════════════════════════════════════════════════════════════════════

class ConfusionMatrixVisualizer:
    """Seaborn confusion matrices: one subplot per (class × method) combination.

    Saves to: eval_confusion_matrix.png
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def plot(
        self,
        summary: Dict[str, Any],
        tumor_classes: List[str],
        methods: List[str],
        filename: str = "eval_confusion_matrix.png",
    ) -> str:
        """Render and save confusion matrices.

        Parameters
        ----------
        summary       : Nested dict  summary[method][class]["cm"] → 2×2 array-like
                        [[TN, FP], [FN, TP]].
        tumor_classes : List of tumor class names.
        methods       : List of method names.
        filename      : Output filename.
        """
        try:
            import seaborn as sns
        except ImportError:
            sns = None

        n_rows = len(tumor_classes)
        n_cols = len(methods)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 3.5 * n_rows),
            squeeze=False,
        )
        _apply_dark_theme(fig, axes)

        class_labels = ["Non-tumor", "Tumor"]

        for r, cls in enumerate(tumor_classes):
            for c, method in enumerate(methods):
                ax = axes[r][c]
                cm_raw = summary.get(method, {}).get(cls, {}).get("cm", None)
                if cm_raw is None:
                    cm = np.zeros((2, 2), dtype=int)
                else:
                    cm = np.array(cm_raw, dtype=int)

                if sns is not None:
                    sns.heatmap(
                        cm, ax=ax,
                        annot=True, fmt="d",
                        annot_kws={"size": 10, "color": _WHITE},
                        xticklabels=class_labels,
                        yticklabels=class_labels,
                        cmap="Blues",
                        linewidths=0.5, linecolor=_SPINE,
                        cbar=False,
                    )
                else:
                    im = ax.imshow(cm, cmap="Blues", aspect="auto")
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(class_labels, fontsize=8)
                    ax.set_yticks([0, 1])
                    ax.set_yticklabels(class_labels, fontsize=8)
                    for ri in range(2):
                        for ci in range(2):
                            ax.text(ci, ri, str(cm[ri, ci]),
                                    ha="center", va="center",
                                    color=_WHITE, fontsize=10)

                ax.set_title(
                    f"{method} — {cls.capitalize()}",
                    color=_WHITE, fontsize=9, fontweight="bold",
                )
                ax.set_xlabel("Predicted", color=_WHITE, fontsize=8)
                ax.set_ylabel("Actual", color=_WHITE, fontsize=8)
                ax.tick_params(colors=_WHITE, labelsize=8)

        fig.suptitle(
            "Confusion Matrices per Class × Method",
            color=_WHITE, fontsize=13, fontweight="bold",
        )
        fig.tight_layout()

        out = os.path.join(self.cfg.output_dir, filename)
        _save(fig, out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 9. AblationVisualizer
# ══════════════════════════════════════════════════════════════════════════════

class AblationVisualizer:
    """Ablation study: WithEdge vs WithoutEdge bar + delta heatmap.

    Saves to: eval_ablasi_edge.png
    """

    _METRICS = ["fom", "f1", "dice", "iou"]
    _METRIC_LABELS = ["FOM", "F1", "Dice", "IoU"]

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def plot(
        self,
        smm_abl: Dict[str, Any],
        tumor_classes: List[str],
        filename: str = "eval_ablasi_edge.png",
    ) -> str:
        """Render and save the ablation figure.

        Parameters
        ----------
        smm_abl       : Dict with keys "with_edge" and "without_edge", each
                        mapping  smm_abl[variant][class][metric] → float.
        tumor_classes : List of tumor class names.
        filename      : Output filename.
        """
        try:
            import seaborn as sns
        except ImportError:
            sns = None

        variants = ["with_edge", "without_edge"]
        var_labels = ["With Edge (HQEHED-AMT)", "Without Edge (Canny)"]
        var_colors = [_PALETTE["HQEHED-AMT"], _PALETTE["Canny"]]

        n_classes = len(tumor_classes)
        # Layout: top row = grouped bars per class, bottom row = delta heatmap
        fig = plt.figure(figsize=(6 * n_classes, 9))
        fig.patch.set_facecolor(_BG)

        bar_axes   = []
        delta_axes = []

        for i, cls in enumerate(tumor_classes):
            ax_bar = fig.add_subplot(2, n_classes, i + 1)
            ax_bar.set_facecolor(_AX)
            bar_axes.append(ax_bar)

            ax_delta = fig.add_subplot(2, n_classes, n_classes + i + 1)
            ax_delta.set_facecolor(_AX)
            delta_axes.append(ax_delta)

        _apply_dark_theme(fig, bar_axes + delta_axes)

        for i, cls in enumerate(tumor_classes):
            ax_bar = bar_axes[i]
            x      = np.arange(len(self._METRICS))
            width  = 0.35

            for j, (var, lbl, col) in enumerate(zip(variants, var_labels, var_colors)):
                vals = [
                    float(smm_abl.get(var, {}).get(cls, {}).get(m, 0.0))
                    for m in self._METRICS
                ]
                offset = (j - 0.5) * width
                bars   = ax_bar.bar(x + offset, vals, width=width * 0.9,
                                    color=col, label=lbl, alpha=0.85)
                for bar, val in zip(bars, vals):
                    ax_bar.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom",
                        color=_WHITE, fontsize=6.5, rotation=90,
                    )

            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(self._METRIC_LABELS, color=_WHITE, fontsize=9)
            ax_bar.set_title(cls.capitalize(), color=_WHITE, fontsize=11, fontweight="bold")
            ax_bar.set_ylim(0, 1.2)
            ax_bar.set_ylabel("Score", color=_WHITE, fontsize=9)
            ax_bar.grid(True, axis="y", color=_GRID, linestyle="--", linewidth=0.6, alpha=0.7)
            ax_bar.legend(facecolor=_AX, edgecolor=_SPINE, labelcolor=_WHITE, fontsize=7)

            # Delta heatmap (with_edge − without_edge)
            ax_delta = delta_axes[i]
            delta = np.array([[
                float(smm_abl.get("with_edge", {}).get(cls, {}).get(m, 0.0))
                - float(smm_abl.get("without_edge", {}).get(cls, {}).get(m, 0.0))
                for m in self._METRICS
            ]])  # shape (1, n_metrics)

            if sns is not None:
                sns.heatmap(
                    delta, ax=ax_delta,
                    annot=True, fmt="+.3f",
                    annot_kws={"size": 9, "color": _WHITE},
                    xticklabels=self._METRIC_LABELS,
                    yticklabels=["Δ (With − Without)"],
                    cmap="RdYlGn", center=0.0,
                    vmin=-0.2, vmax=0.2,
                    linewidths=0.5, linecolor=_SPINE,
                    cbar_kws={"shrink": 0.6},
                )
                cbar = ax_delta.collections[0].colorbar
                cbar.ax.yaxis.set_tick_params(color=_WHITE)
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_WHITE, fontsize=7)
            else:
                im = ax_delta.imshow(delta, cmap="RdYlGn", aspect="auto",
                                     vmin=-0.2, vmax=0.2)
                ax_delta.set_xticks(range(len(self._METRIC_LABELS)))
                ax_delta.set_xticklabels(self._METRIC_LABELS, fontsize=8)
                ax_delta.set_yticks([0])
                ax_delta.set_yticklabels(["Δ"], fontsize=8)
                for ci, val in enumerate(delta[0]):
                    ax_delta.text(ci, 0, f"{val:+.3f}",
                                  ha="center", va="center",
                                  color=_WHITE, fontsize=9)
                fig.colorbar(im, ax=ax_delta, fraction=0.046, pad=0.04)

            ax_delta.set_title(
                f"Δ Metrics — {cls.capitalize()}",
                color=_WHITE, fontsize=10, fontweight="bold",
            )
            ax_delta.tick_params(colors=_WHITE, labelsize=8)

        fig.suptitle(
            "Ablation Study: With Edge vs Without Edge",
            color=_WHITE, fontsize=13, fontweight="bold",
        )
        fig.tight_layout()

        out = os.path.join(self.cfg.output_dir, filename)
        _save(fig, out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 10. SpeedBenchmarkVisualizer
# ══════════════════════════════════════════════════════════════════════════════

class SpeedBenchmarkVisualizer:
    """Execution time bar + FPS bar + speedup bar for method benchmarking.

    Saves to: eval_speed_benchmark.png
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def plot(
        self,
        spd_st: Dict[str, Any],
        methods: List[str],
        filename: str = "eval_speed_benchmark.png",
    ) -> str:
        """Render and save the speed benchmark figure.

        Parameters
        ----------
        spd_st  : Dict  spd_st[method] → {"mean": float, "std": float,
                                           "fps": float, "speedup": float}.
                  All time values in seconds per image.
        methods : List of method names (defines bar order).
        filename: Output filename.
        """
        means   = [float(spd_st.get(m, {}).get("mean",    0.0)) for m in methods]
        stds    = [float(spd_st.get(m, {}).get("std",     0.0)) for m in methods]
        fps_vals = [float(spd_st.get(m, {}).get("fps",    0.0)) for m in methods]
        speedups = [float(spd_st.get(m, {}).get("speedup", 1.0)) for m in methods]
        colors   = [_method_color(m, i) for i, m in enumerate(methods)]

        x = np.arange(len(methods))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        _apply_dark_theme(fig, axes)

        # ── Panel 0: Mean execution time ──────────────────────────────────────
        ax0 = axes[0]
        bars = ax0.bar(x, means, yerr=stds, capsize=4,
                       color=colors, alpha=0.85,
                       error_kw={"ecolor": _WHITE, "elinewidth": 1.2})
        for bar, val, std in zip(bars, means, stds):
            ax0.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.001,
                f"{val:.4f}s", ha="center", va="bottom",
                color=_WHITE, fontsize=8,
            )
        ax0.set_xticks(x)
        ax0.set_xticklabels(methods, color=_WHITE, fontsize=9, rotation=15, ha="right")
        ax0.set_ylabel("Time per Image (s)", color=_WHITE, fontsize=9)
        ax0.set_title("Execution Time (mean ± std)", color=_WHITE,
                      fontsize=10, fontweight="bold")
        ax0.grid(True, axis="y", color=_GRID, linestyle="--", linewidth=0.6, alpha=0.7)

        # ── Panel 1: FPS ──────────────────────────────────────────────────────
        ax1 = axes[1]
        bars1 = ax1.bar(x, fps_vals, color=colors, alpha=0.85)
        for bar, val in zip(bars1, fps_vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom",
                color=_WHITE, fontsize=8,
            )
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, color=_WHITE, fontsize=9, rotation=15, ha="right")
        ax1.set_ylabel("Frames per Second (FPS)", color=_WHITE, fontsize=9)
        ax1.set_title("Throughput (FPS)", color=_WHITE, fontsize=10, fontweight="bold")
        ax1.grid(True, axis="y", color=_GRID, linestyle="--", linewidth=0.6, alpha=0.7)

        # ── Panel 2: Speedup relative to baseline ─────────────────────────────
        ax2 = axes[2]
        bars2 = ax2.bar(x, speedups, color=colors, alpha=0.85)
        ax2.axhline(1.0, color=_WHITE, linestyle="--", linewidth=0.8, alpha=0.6)
        for bar, val in zip(bars2, speedups):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}×", ha="center", va="bottom",
                color=_WHITE, fontsize=8,
            )
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, color=_WHITE, fontsize=9, rotation=15, ha="right")
        ax2.set_ylabel("Speedup (×)", color=_WHITE, fontsize=9)
        ax2.set_title("Speedup vs Baseline", color=_WHITE, fontsize=10, fontweight="bold")
        ax2.grid(True, axis="y", color=_GRID, linestyle="--", linewidth=0.6, alpha=0.7)

        fig.suptitle(
            "Speed Benchmark: Execution Time / FPS / Speedup",
            color=_WHITE, fontsize=13, fontweight="bold",
        )
        fig.tight_layout()

        out = os.path.join(self.cfg.output_dir, filename)
        _save(fig, out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 11. DomainValidationVisualizer
# ══════════════════════════════════════════════════════════════════════════════

class DomainValidationVisualizer:
    """FOM boxplot + bar chart comparing natural-image vs MRI domain.

    Saves to:
      eval_domain_fom_boxplot.png
      eval_domain_fom_bar.png
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def plot(
        self,
        res_digit: Dict[str, Any],
        res_mri: Dict[str, Any],
        methods: List[str],
        filename_box: str = "eval_domain_fom_boxplot.png",
        filename_bar: str = "eval_domain_fom_bar.png",
    ) -> Tuple[str, str]:
        """Render and save the domain validation figures.

        Parameters
        ----------
        res_digit   : Dict  res_digit[method] → list of FOM values (natural images).
        res_mri     : Dict  res_mri[method]   → list of FOM values (MRI images).
        methods     : List of method names.
        filename_box: Output filename for the boxplot figure.
        filename_bar: Output filename for the bar figure.

        Returns
        -------
        Tuple[str, str] : Paths to the saved boxplot and bar figures.
        """
        colors = [_method_color(m, i) for i, m in enumerate(methods)]

        # ── Figure 1: Boxplot ─────────────────────────────────────────────────
        fig_box, axes_box = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        _apply_dark_theme(fig_box, axes_box)

        for ax, res, domain_label in zip(
            axes_box,
            [res_digit, res_mri],
            ["Natural Images (BSDS500)", "MRI Images"],
        ):
            data_list = [res.get(m, []) for m in methods]
            bp = ax.boxplot(
                data_list,
                patch_artist=True,
                medianprops={"color": _WHITE, "linewidth": 1.5},
                whiskerprops={"color": _WHITE},
                capprops={"color": _WHITE},
                flierprops={"markerfacecolor": _WHITE, "markersize": 3,
                            "markeredgecolor": _WHITE},
            )
            for patch, col in zip(bp["boxes"], colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.75)

            ax.set_xticks(range(1, len(methods) + 1))
            ax.set_xticklabels(methods, color=_WHITE, fontsize=9, rotation=15, ha="right")
            ax.set_ylabel("FOM Score", color=_WHITE, fontsize=9)
            ax.set_title(domain_label, color=_WHITE, fontsize=10, fontweight="bold")
            ax.grid(True, axis="y", color=_GRID, linestyle="--", linewidth=0.6, alpha=0.7)

        fig_box.suptitle(
            "Domain Validation: FOM Distribution (Natural vs MRI)",
            color=_WHITE, fontsize=13, fontweight="bold",
        )
        fig_box.tight_layout()
        out_box = os.path.join(self.cfg.output_dir, filename_box)
        _save(fig_box, out_box)

        # ── Figure 2: Bar chart (mean FOM) ────────────────────────────────────
        fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
        _apply_dark_theme(fig_bar, ax_bar)

        x      = np.arange(len(methods))
        width  = 0.35
        means_digit = [float(np.mean(res_digit.get(m, [0.0]))) for m in methods]
        means_mri   = [float(np.mean(res_mri.get(m,   [0.0]))) for m in methods]
        stds_digit  = [float(np.std(res_digit.get(m,  [0.0]))) for m in methods]
        stds_mri    = [float(np.std(res_mri.get(m,    [0.0]))) for m in methods]

        bars_d = ax_bar.bar(
            x - width / 2, means_digit, width=width * 0.9,
            yerr=stds_digit, capsize=4,
            color="#58a6ff", alpha=0.85, label="Natural Images",
            error_kw={"ecolor": _WHITE, "elinewidth": 1.0},
        )
        bars_m = ax_bar.bar(
            x + width / 2, means_mri, width=width * 0.9,
            yerr=stds_mri, capsize=4,
            color="#f78166", alpha=0.85, label="MRI Images",
            error_kw={"ecolor": _WHITE, "elinewidth": 1.0},
        )

        for bars, stds in [(bars_d, stds_digit), (bars_m, stds_mri)]:
            for bar, std in zip(bars, stds):
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", color=_WHITE, fontsize=7.5,
                )

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(methods, color=_WHITE, fontsize=9, rotation=15, ha="right")
        ax_bar.set_ylabel("Mean FOM Score", color=_WHITE, fontsize=10)
        ax_bar.set_title(
            "Domain Validation: Mean FOM — Natural vs MRI",
            color=_WHITE, fontsize=12, fontweight="bold",
        )
        ax_bar.set_ylim(0, 1.1)
        ax_bar.grid(True, axis="y", color=_GRID, linestyle="--", linewidth=0.6, alpha=0.7)
        ax_bar.legend(facecolor=_AX, edgecolor=_SPINE, labelcolor=_WHITE, fontsize=9)

        fig_bar.tight_layout()
        out_bar = os.path.join(self.cfg.output_dir, filename_bar)
        _save(fig_bar, out_bar)

        return out_box, out_bar


# ── Public API ─────────────────────────────────────────────────────────────────
__all__ = [
    "PanelVisualizer",
    "ErrorMapVisualizer",
    "PSOConvergenceVisualizer",
    "MetricsBarVisualizer",
    "PSOAnalysisVisualizer",
    "ComparativeBarVisualizer",
    "SegmentationHeatmapVisualizer",
    "ConfusionMatrixVisualizer",
    "AblationVisualizer",
    "SpeedBenchmarkVisualizer",
    "DomainValidationVisualizer",
]
