"""Core modules: Config, DatasetLoader, Preprocessor, edge detectors."""

from .config import Config
from .dataset import DatasetLoader, Preprocessor
from .quantum_edge import (
    HQEHEDPipeline,
    QPIEEncoder,
    HadamardEdgeAmplitude,
    AdaptiveMeanThreshold,
    PPQECCorrector,
)
from .classic_edge import (
    EdgeDetector,
    CannyDetector,
    SobelDetector,
    PrewittDetector,
    LoGDetector,
    EdgeDetectorFactory,
)

__all__ = [
    "Config",
    "DatasetLoader",
    "Preprocessor",
    "HQEHEDPipeline",
    "QPIEEncoder",
    "HadamardEdgeAmplitude",
    "AdaptiveMeanThreshold",
    "PPQECCorrector",
    "EdgeDetector",
    "CannyDetector",
    "SobelDetector",
    "PrewittDetector",
    "LoGDetector",
    "EdgeDetectorFactory",
]
