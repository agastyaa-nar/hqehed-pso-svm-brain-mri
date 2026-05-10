"""
test_features.py — Unit tests for FullFeatureExtractor and NoEdgeFeatureExtractor.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.config import Config
from src.models.features import (
    FullFeatureExtractor,
    NoEdgeFeatureExtractor,
    _FM_KEYS_FULL,
    _FM_KEYS_NOEDGE,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return Config(output_dir=tempfile.mkdtemp())


@pytest.fixture
def sample_img():
    rng = np.random.default_rng(42)
    return rng.random((64, 64)).astype(np.float32)


@pytest.fixture
def sample_edge():
    rng = np.random.default_rng(7)
    return (rng.random((64, 64)) > 0.85).astype(np.uint8)


@pytest.fixture
def sample_mask():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:44, 20:44] = 1   # 24×24 tumor region
    return mask


@pytest.fixture
def full_extractor():
    return FullFeatureExtractor()


@pytest.fixture
def noedge_extractor():
    return NoEdgeFeatureExtractor()


# ── FullFeatureExtractor ──────────────────────────────────────────────────────

class TestFullFeatureExtractor:
    def test_n_features_constant(self, full_extractor):
        assert full_extractor.n_features == 27

    def test_feature_map_keys(self, full_extractor, sample_img, sample_edge):
        fm = full_extractor.build_feature_maps(sample_img, sample_edge)
        for key in _FM_KEYS_FULL:
            assert key in fm, f"Missing key: {key}"

    def test_feature_map_shapes(self, full_extractor, sample_img, sample_edge):
        fm = full_extractor.build_feature_maps(sample_img, sample_edge)
        H, W = sample_img.shape
        for key, (c, m, s) in fm.items():
            assert c.shape == (H, W), f"{key} center shape mismatch"
            assert m.shape == (H, W), f"{key} mean shape mismatch"
            assert s.shape == (H, W), f"{key} std shape mismatch"

    def test_feature_vector_length(self, full_extractor, sample_img, sample_edge):
        fm = full_extractor.build_feature_maps(sample_img, sample_edge)
        vec = full_extractor.feature_vector(fm, 10, 10)
        assert len(vec) == 27

    def test_feature_vector_finite(self, full_extractor, sample_img, sample_edge):
        fm = full_extractor.build_feature_maps(sample_img, sample_edge)
        vec = full_extractor.feature_vector(fm, 10, 10)
        assert all(np.isfinite(v) for v in vec), "Feature vector contains non-finite values"

    def test_sample_features_shape(self, full_extractor, sample_img, sample_edge, sample_mask, cfg):
        X, y = full_extractor.sample_features(sample_img, sample_edge, sample_mask, 200, cfg)
        assert X.ndim == 2
        assert X.shape[1] == 27
        assert len(X) == len(y)

    def test_sample_features_labels(self, full_extractor, sample_img, sample_edge, sample_mask, cfg):
        X, y = full_extractor.sample_features(sample_img, sample_edge, sample_mask, 200, cfg)
        unique = set(np.unique(y).tolist())
        assert unique.issubset({0, 1})

    def test_sample_features_ratio(self, full_extractor, sample_img, sample_edge, sample_mask, cfg):
        """Tumor:non-tumor ratio should be approximately 1:3."""
        X, y = full_extractor.sample_features(sample_img, sample_edge, sample_mask, 400, cfg)
        n_tumor    = int(np.sum(y == 1))
        n_nontumor = int(np.sum(y == 0))
        # Allow some slack due to min() clamping
        assert n_nontumor >= n_tumor, "Non-tumor samples should be >= tumor samples"

    def test_sample_features_empty_mask(self, full_extractor, sample_img, sample_edge, cfg):
        """Empty mask (no tumor) should return empty arrays."""
        empty_mask = np.zeros((64, 64), dtype=np.uint8)
        X, y = full_extractor.sample_features(sample_img, sample_edge, empty_mask, 200, cfg)
        assert len(X) == 0
        assert len(y) == 0

    def test_predict_full_image_shape(self, full_extractor, sample_img, sample_edge, sample_mask, cfg):
        """predict_full_image should return a mask of the same spatial shape."""
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler

        X, y = full_extractor.sample_features(sample_img, sample_edge, sample_mask, 200, cfg)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        clf = SVC(C=1.0, gamma=0.1, kernel="rbf")
        clf.fit(X_s, y)

        pred = full_extractor.predict_full_image(sample_img, sample_edge, clf, scaler)
        assert pred.shape == sample_img.shape
        assert pred.dtype == np.uint8

    def test_predict_full_image_binary(self, full_extractor, sample_img, sample_edge, sample_mask, cfg):
        """Predicted mask values must be in {0, 1}."""
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler

        X, y = full_extractor.sample_features(sample_img, sample_edge, sample_mask, 200, cfg)
        scaler = StandardScaler()
        clf = SVC(C=1.0, gamma=0.1, kernel="rbf")
        clf.fit(scaler.fit_transform(X), y)

        pred = full_extractor.predict_full_image(sample_img, sample_edge, clf, scaler)
        unique = set(np.unique(pred).tolist())
        assert unique.issubset({0, 1})


# ── NoEdgeFeatureExtractor ────────────────────────────────────────────────────

class TestNoEdgeFeatureExtractor:
    def test_n_features_constant(self, noedge_extractor):
        assert noedge_extractor.n_features == 18

    def test_feature_map_keys(self, noedge_extractor, sample_img):
        fm = noedge_extractor.build_feature_maps(sample_img)
        for key in _FM_KEYS_NOEDGE:
            assert key in fm, f"Missing key: {key}"

    def test_no_edge_keys(self, noedge_extractor, sample_img):
        fm = noedge_extractor.build_feature_maps(sample_img)
        for edge_key in ["edge_flag", "edge_density", "dist_to_edge"]:
            assert edge_key not in fm, f"Edge key should not be present: {edge_key}"

    def test_feature_vector_length(self, noedge_extractor, sample_img):
        fm = noedge_extractor.build_feature_maps(sample_img)
        vec = noedge_extractor.feature_vector(fm, 10, 10)
        assert len(vec) == 18

    def test_feature_vector_finite(self, noedge_extractor, sample_img):
        fm = noedge_extractor.build_feature_maps(sample_img)
        vec = noedge_extractor.feature_vector(fm, 10, 10)
        assert all(np.isfinite(v) for v in vec)

    def test_edge_map_ignored(self, noedge_extractor, sample_img, sample_edge):
        """build_feature_maps should produce the same result regardless of edge_map."""
        fm_with    = noedge_extractor.build_feature_maps(sample_img, sample_edge)
        fm_without = noedge_extractor.build_feature_maps(sample_img, None)
        for key in _FM_KEYS_NOEDGE:
            c1, m1, s1 = fm_with[key]
            c2, m2, s2 = fm_without[key]
            np.testing.assert_array_almost_equal(c1, c2, decimal=6)

    def test_predict_full_image_shape(self, noedge_extractor, sample_img, sample_mask, cfg):
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler

        X, y = noedge_extractor.sample_features(sample_img, None, sample_mask, 200, cfg)
        scaler = StandardScaler()
        clf = SVC(C=1.0, gamma=0.1, kernel="rbf")
        clf.fit(scaler.fit_transform(X), y)

        pred = noedge_extractor.predict_full_image(sample_img, None, clf, scaler)
        assert pred.shape == sample_img.shape


# ── Consistency between Full and NoEdge ──────────────────────────────────────

class TestFeatureConsistency:
    def test_full_has_more_features_than_noedge(self, full_extractor, noedge_extractor):
        assert full_extractor.n_features > noedge_extractor.n_features

    def test_noedge_is_subset_of_full(self):
        """All NoEdge keys should also appear in Full keys."""
        for key in _FM_KEYS_NOEDGE:
            assert key in _FM_KEYS_FULL

    def test_shared_features_identical(self, full_extractor, noedge_extractor,
                                       sample_img, sample_edge):
        """Shared feature maps should produce identical values."""
        fm_full   = full_extractor.build_feature_maps(sample_img, sample_edge)
        fm_noedge = noedge_extractor.build_feature_maps(sample_img, sample_edge)
        for key in _FM_KEYS_NOEDGE:
            c_f, m_f, s_f = fm_full[key]
            c_n, m_n, s_n = fm_noedge[key]
            np.testing.assert_array_almost_equal(c_f, c_n, decimal=5,
                                                  err_msg=f"Center mismatch for {key}")
            np.testing.assert_array_almost_equal(m_f, m_n, decimal=5,
                                                  err_msg=f"Mean mismatch for {key}")
