"""
config.py — Centralized configuration dataclass for HQEHED-AMT + PSO-SVM.

All hyperparameters live here. Supports YAML/JSON serialization via
to_dict() / from_dict() for reproducible experiments.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional


@dataclass
class Config:
    # ── Dataset ───────────────────────────────────────────────────
    data_root: str = "../dataset"
    output_dir: str = "./results"
    img_size: int = 512
    max_images: Optional[int] = 30

    # ── Preprocessing (Section 3.2) ───────────────────────────────
    gauss_ksize: int = 3        # Gaussian kernel size (must be odd)
    gauss_sigma: float = 0.5    # Gaussian sigma; ↑ = smoother

    # ── HQEHED-AMT (Section 3.3) ──────────────────────────────────
    crop_size: int = 512        # Segment length per scan (rounded to pow2)
    gamma: float = 1.0          # Positive edge threshold multiplier
    gamma_neg: float = 1.0      # Negative edge threshold multiplier

    # ── PSO (Section 3.5, Eq. 2.27–2.28) ─────────────────────────
    pso_particles: int = 20
    pso_iterations: int = 50
    pso_w: float = 0.7          # Inertia weight
    pso_c1: float = 1.5         # Cognitive factor ϖ₁
    pso_c2: float = 1.5         # Social factor   ϖ₂
    pso_c_range: Tuple[float, float] = (-1.0, 3.0)   # log10(C) ∈ [0.1, 1000]
    pso_g_range: Tuple[float, float] = (-3.0, 1.0)   # log10(γ) ∈ [0.001, 10]

    # ── SVM ───────────────────────────────────────────────────────
    svm_kernel: str = "rbf"
    patch_size: int = 5                         # Local patch size for features
    svm_class_weight: dict = field(
        default_factory=lambda: {0: 1, 1: 2}   # Upweight tumor class
    )

    # ── Post-processing ───────────────────────────────────────────
    cc_min_size: int = 1000     # Minimum blob size (pixels) to keep
    cc_max_blobs: int = 2       # Maximum number of tumor regions to retain

    # ── Sampling ──────────────────────────────────────────────────
    n_samples_per_img: int = 800
    eval_n_tumor: int = 1500
    eval_n_nontumor: int = 6000

    # ── Evaluation ────────────────────────────────────────────────
    alpha_fom: float = 0.5      # Pratt's FOM scaling constant (Eq. 2.39)

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

    # ── Serialization ─────────────────────────────────────────────
    def to_dict(self) -> dict:
        """Convert config to a plain dict (JSON/YAML serializable)."""
        d = asdict(self)
        # Convert tuples to lists for YAML compatibility
        d["pso_c_range"] = list(d["pso_c_range"])
        d["pso_g_range"] = list(d["pso_g_range"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Reconstruct Config from a plain dict."""
        d = dict(d)
        if "pso_c_range" in d:
            d["pso_c_range"] = tuple(d["pso_c_range"])
        if "pso_g_range" in d:
            d["pso_g_range"] = tuple(d["pso_g_range"])
        if "svm_class_weight" in d:
            # JSON keys are always strings; convert back to int
            d["svm_class_weight"] = {int(k): v for k, v in d["svm_class_weight"].items()}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    def to_yaml(self, path: str) -> None:
        """Save config to a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required: pip install pyyaml")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        lines = ["Config("]
        for k, v in asdict(self).items():
            lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)
