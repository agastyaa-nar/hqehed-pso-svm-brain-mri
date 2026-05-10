"""
test_dataset.py — Unit tests for Config and Preprocessor.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.config import Config
from src.core.dataset import DatasetLoader, Preprocessor


# ── Config ────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_values(self):
        cfg = Config(output_dir=tempfile.mkdtemp())
        assert cfg.img_size == 512
        assert cfg.gauss_ksize == 3
        assert cfg.gauss_sigma == 0.5
        assert cfg.gamma == 1.0
        assert cfg.gamma_neg == 1.0
        assert cfg.pso_particles == 20
        assert cfg.pso_iterations == 50
        assert cfg.svm_kernel == "rbf"

    def test_output_dir_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "new_results")
            cfg = Config(output_dir=out)
            assert os.path.isdir(out)

    def test_to_dict_roundtrip(self):
        cfg = Config(output_dir=tempfile.mkdtemp(), gamma=0.3, crop_size=64)
        d = cfg.to_dict()
        cfg2 = Config.from_dict(d)
        assert cfg2.gamma == cfg.gamma
        assert cfg2.crop_size == cfg.crop_size
        assert cfg2.img_size == cfg.img_size

    def test_to_dict_pso_ranges_are_lists(self):
        cfg = Config(output_dir=tempfile.mkdtemp())
        d = cfg.to_dict()
        assert isinstance(d["pso_c_range"], list)
        assert isinstance(d["pso_g_range"], list)

    def test_from_dict_pso_ranges_are_tuples(self):
        cfg = Config(output_dir=tempfile.mkdtemp())
        d = cfg.to_dict()
        cfg2 = Config.from_dict(d)
        assert isinstance(cfg2.pso_c_range, tuple)
        assert isinstance(cfg2.pso_g_range, tuple)

    def test_from_dict_svm_class_weight_int_keys(self):
        cfg = Config(output_dir=tempfile.mkdtemp())
        d = cfg.to_dict()
        # Simulate JSON round-trip (keys become strings)
        d["svm_class_weight"] = {"0": 1, "1": 2}
        cfg2 = Config.from_dict(d)
        assert 0 in cfg2.svm_class_weight
        assert 1 in cfg2.svm_class_weight

    def test_yaml_roundtrip(self, tmp_path):
        pytest.importorskip("yaml")
        cfg = Config(output_dir=str(tmp_path / "results"), gamma=0.25)
        yaml_path = str(tmp_path / "config.yaml")
        cfg.to_yaml(yaml_path)
        cfg2 = Config.from_yaml(yaml_path)
        assert abs(cfg2.gamma - 0.25) < 1e-9

    def test_repr_contains_gamma(self):
        cfg = Config(output_dir=tempfile.mkdtemp(), gamma=0.42)
        r = repr(cfg)
        assert "gamma" in r


# ── Preprocessor ─────────────────────────────────────────────────────────────

class TestPreprocessor:
    @pytest.fixture
    def cfg(self):
        return Config(output_dir=tempfile.mkdtemp())

    @pytest.fixture
    def prep(self, cfg):
        return Preprocessor(cfg)

    @pytest.fixture
    def sample_img(self):
        rng = np.random.default_rng(0)
        return rng.random((64, 64)).astype(np.float32)

    def test_output_shape_unchanged(self, prep, sample_img):
        out = prep.process(sample_img)
        assert out.shape == sample_img.shape

    def test_output_dtype_float32(self, prep, sample_img):
        out = prep.process(sample_img)
        assert out.dtype == np.float32

    def test_blur_reduces_variance(self, prep, sample_img):
        """Gaussian blur should reduce pixel variance."""
        out = prep.process(sample_img, ksize=7, sigma=2.0)
        assert float(np.var(out)) < float(np.var(sample_img))

    def test_even_ksize_corrected(self, prep, sample_img):
        """Even ksize should be auto-corrected to odd without raising."""
        out = prep.process(sample_img, ksize=4)
        assert out.shape == sample_img.shape

    def test_sigma_override(self, prep, sample_img):
        """Higher sigma → more blur → lower variance."""
        out_low  = prep.process(sample_img, sigma=0.3)
        out_high = prep.process(sample_img, sigma=2.0)
        assert float(np.var(out_high)) < float(np.var(out_low))


# ── DatasetLoader (filesystem-based) ─────────────────────────────────────────

class TestDatasetLoader:
    @pytest.fixture
    def fake_dataset(self, tmp_path):
        """Create a minimal fake dataset structure."""
        for tumor_type in ["glioma", "meningioma"]:
            img_dir  = tmp_path / tumor_type / "images"
            mask_dir = tmp_path / tumor_type / "masks"
            img_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            for i in range(3):
                fname = f"img_{i:03d}.png"
                img  = (np.random.rand(32, 32) * 255).astype(np.uint8)
                mask = (np.random.rand(32, 32) * 255).astype(np.uint8)
                cv2.imwrite(str(img_dir  / fname), img)
                cv2.imwrite(str(mask_dir / fname), mask)
        return tmp_path

    @pytest.fixture
    def loader(self, fake_dataset):
        cfg = Config(data_root=str(fake_dataset), output_dir=tempfile.mkdtemp())
        return DatasetLoader(cfg)

    def test_load_pairs_count(self, loader):
        pairs = loader.load_pairs()
        assert len(pairs) == 6   # 2 classes × 3 images

    def test_load_pairs_by_class(self, loader):
        grouped = loader.load_pairs_by_class()
        assert "glioma" in grouped
        assert "meningioma" in grouped
        assert len(grouped["glioma"]) == 3

    def test_load_pairs_by_class_filter(self, loader):
        grouped = loader.load_pairs_by_class(tumor_classes=["glioma"])
        assert "meningioma" not in grouped

    def test_load_image_mask_shapes(self, loader):
        pairs = loader.load_pairs()
        img, mask = loader.load_image_mask(pairs[0][0], pairs[0][1], size=64)
        assert img.shape  == (64, 64)
        assert mask.shape == (64, 64)

    def test_load_image_dtype(self, loader):
        pairs = loader.load_pairs()
        img, mask = loader.load_image_mask(pairs[0][0], pairs[0][1])
        assert img.dtype  == np.float32
        assert mask.dtype == np.uint8

    def test_load_image_range(self, loader):
        pairs = loader.load_pairs()
        img, mask = loader.load_image_mask(pairs[0][0], pairs[0][1])
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_load_mask_binary(self, loader):
        pairs = loader.load_pairs()
        _, mask = loader.load_image_mask(pairs[0][0], pairs[0][1])
        unique = set(np.unique(mask).tolist())
        assert unique.issubset({0, 1})

    def test_available_classes(self, loader):
        classes = loader.available_classes()
        assert "glioma" in classes
        assert "meningioma" in classes

    def test_missing_file_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_image_mask("/nonexistent/img.png", "/nonexistent/mask.png")
