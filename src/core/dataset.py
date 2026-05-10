"""
dataset.py — Dataset loading and preprocessing utilities.

Classes:
    DatasetLoader  — scan image/mask pairs, group by tumor class
    Preprocessor   — Gaussian blur noise reduction (Section 3.2)
"""

from __future__ import annotations
import glob
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import Config

# Supported image extensions
_IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")


class DatasetLoader:
    """Scan and load brain MRI image-mask pairs.

    Expected dataset structure::

        <data_root>/
            <tumor_type>/
                images/
                    <filename>.<ext>
                masks/
                    <filename>.<ext>   ← same filename as image

    Tumor types (sub-folders) are inferred automatically.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ── Public API ────────────────────────────────────────────────

    def load_pairs(self) -> List[Tuple[str, str]]:
        """Return all (image_path, mask_path) pairs found under data_root."""
        pairs: List[Tuple[str, str]] = []
        for ext in _IMG_EXTS:
            img_paths = sorted(
                glob.glob(os.path.join(self.cfg.data_root, "*", "images", ext))
            )
            for img in img_paths:
                fname = os.path.basename(img)
                tumor_dir = os.path.dirname(os.path.dirname(img))
                mask = os.path.join(tumor_dir, "masks", fname)
                if os.path.exists(mask) and (img, mask) not in pairs:
                    pairs.append((img, mask))
        return pairs

    def load_pairs_by_class(
        self,
        tumor_classes: Optional[List[str]] = None,
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Return pairs grouped by tumor class (sub-folder name).

        Args:
            tumor_classes: If given, only include these class names.
                           Defaults to all classes found in data_root.

        Returns:
            dict mapping class_name → list of (image_path, mask_path)
        """
        all_pairs = self.load_pairs()
        grouped: Dict[str, List[Tuple[str, str]]] = {}
        for img_p, msk_p in all_pairs:
            tc = os.path.basename(os.path.dirname(os.path.dirname(img_p)))
            if tumor_classes is not None and tc not in tumor_classes:
                continue
            grouped.setdefault(tc, []).append((img_p, msk_p))
        return grouped

    def load_image_mask(
        self,
        img_path: str,
        mask_path: str,
        size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single (image, mask) pair.

        Args:
            img_path  : path to grayscale MRI image
            mask_path : path to binary mask
            size      : resize target (square); defaults to cfg.img_size

        Returns:
            img  : float32 array in [0, 1], shape (size, size)
            mask : uint8 binary array {0, 1}, shape (size, size)

        Raises:
            FileNotFoundError: if either file cannot be read.
        """
        size = size or self.cfg.img_size
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.uint8)
        return img, mask

    # ── Convenience ───────────────────────────────────────────────

    def available_classes(self) -> List[str]:
        """Return sorted list of tumor class names found in data_root."""
        pattern = os.path.join(self.cfg.data_root, "*", "images")
        dirs = glob.glob(pattern)
        return sorted({os.path.basename(os.path.dirname(d)) for d in dirs})


class Preprocessor:
    """Apply Gaussian blur for noise reduction (Section 3.2).

    The blur parameters are taken from Config but can be overridden
    per call for grid-search experiments.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def process(
        self,
        img: np.ndarray,
        ksize: Optional[int] = None,
        sigma: Optional[float] = None,
    ) -> np.ndarray:
        """Apply Gaussian blur to a float32 image in [0, 1].

        Args:
            img   : input image, float32, shape (H, W)
            ksize : kernel size (odd integer); defaults to cfg.gauss_ksize
            sigma : Gaussian sigma; defaults to cfg.gauss_sigma

        Returns:
            Blurred float32 image, same shape as input.
        """
        ksize = ksize if ksize is not None else self.cfg.gauss_ksize
        sigma = sigma if sigma is not None else self.cfg.gauss_sigma
        # Ensure ksize is odd and positive
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(img, (ksize, ksize), sigma).astype(np.float32)
