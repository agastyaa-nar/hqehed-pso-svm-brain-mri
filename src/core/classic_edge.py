"""
classic_edge.py — Classic edge detectors used as comparison baselines.

Classes:
    EdgeDetector         — Abstract base class with unified detect() interface
    CannyDetector        — Canny edge detector (OpenCV)
    SobelDetector        — Sobel gradient magnitude + Otsu threshold
    PrewittDetector      — Prewitt gradient magnitude + Otsu threshold
    LoGDetector          — Laplacian of Gaussian (zero-crossing)
    EdgeDetectorFactory  — Registry for creating detectors by name
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import cv2
import numpy as np


class EdgeDetector(ABC):
    """Abstract base class for all edge detectors.

    Subclasses must implement :meth:`detect`.
    """

    @abstractmethod
    def detect(self, img_pre: np.ndarray) -> np.ndarray:
        """Detect edges in a preprocessed grayscale image.

        Args:
            img_pre : float32 image in [0, 1], shape (H, W)

        Returns:
            edge_map : uint8 binary array {0, 1}, shape (H, W)
        """

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        """Convert float32 [0,1] image to uint8 [0,255]."""
        return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

    @staticmethod
    def _otsu_binary(mag: np.ndarray) -> np.ndarray:
        """Apply Otsu threshold to a magnitude image; return binary {0,1}."""
        mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, edge = cv2.threshold(mag_u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return edge.astype(np.uint8)


class CannyDetector(EdgeDetector):
    """Canny edge detector.

    Args:
        low  : lower hysteresis threshold (default 50)
        high : upper hysteresis threshold (default 150)
    """

    def __init__(self, low: int = 50, high: int = 150):
        self.low = low
        self.high = high

    def detect(self, img_pre: np.ndarray) -> np.ndarray:
        img_u8 = self._to_uint8(img_pre)
        return (cv2.Canny(img_u8, self.low, self.high) // 255).astype(np.uint8)


class SobelDetector(EdgeDetector):
    """Sobel gradient magnitude with Otsu thresholding."""

    def detect(self, img_pre: np.ndarray) -> np.ndarray:
        img_u8 = self._to_uint8(img_pre)
        gx = cv2.Sobel(img_u8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_u8, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        return self._otsu_binary(mag)


class PrewittDetector(EdgeDetector):
    """Prewitt gradient magnitude with Otsu thresholding."""

    def detect(self, img_pre: np.ndarray) -> np.ndarray:
        img_f = np.clip(img_pre, 0.0, 1.0).astype(np.float32)
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32) / 3.0
        gx = cv2.filter2D(img_f, cv2.CV_32F, kx)
        gy = cv2.filter2D(img_f, cv2.CV_32F, kx.T)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        return self._otsu_binary(mag)


class LoGDetector(EdgeDetector):
    """Laplacian of Gaussian (zero-crossing) edge detector.

    Args:
        sigma : Gaussian sigma for pre-smoothing (default 1.4)
    """

    def __init__(self, sigma: float = 1.4):
        self.sigma = sigma

    def detect(self, img_pre: np.ndarray) -> np.ndarray:
        img_u8 = self._to_uint8(img_pre)
        ks = int(6 * self.sigma + 1) | 1   # ensure odd
        blur = cv2.GaussianBlur(img_u8, (ks, ks), self.sigma)
        lap = cv2.Laplacian(blur.astype(np.float32), cv2.CV_32F)
        k3 = np.ones((3, 3), np.uint8)
        zc = np.logical_and(
            cv2.dilate((lap > 0).astype(np.uint8), k3),
            cv2.dilate((lap <= 0).astype(np.uint8), k3),
        ).astype(np.uint8)
        return zc


class EdgeDetectorFactory:
    """Registry for creating edge detectors by name.

    Built-in names: ``"canny"``, ``"sobel"``, ``"prewitt"``, ``"log"``

    Usage::

        factory = EdgeDetectorFactory()
        detector = factory.create("canny")
        edge_map = detector.detect(img_pre)

        # Register a custom detector
        factory.register("my_detector", MyDetector)
    """

    _registry: Dict[str, Type[EdgeDetector]] = {
        "canny":   CannyDetector,
        "sobel":   SobelDetector,
        "prewitt": PrewittDetector,
        "log":     LoGDetector,
    }

    def register(self, name: str, cls: Type[EdgeDetector]) -> None:
        """Register a new detector class under the given name."""
        self._registry[name.lower()] = cls

    def create(self, name: str, **kwargs) -> EdgeDetector:
        """Instantiate a detector by name.

        Args:
            name   : detector name (case-insensitive)
            kwargs : forwarded to the detector constructor

        Raises:
            KeyError: if name is not registered
        """
        key = name.lower()
        if key not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"Unknown detector '{name}'. Available: {available}"
            )
        return self._registry[key](**kwargs)

    def available(self) -> list:
        """Return list of registered detector names."""
        return list(self._registry.keys())
