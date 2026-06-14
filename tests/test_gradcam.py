"""
CardioVision — Grad-CAM Unit Tests
Tests for the 1D Grad-CAM explainability module.
"""

import numpy as np
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BEAT_WINDOW, NUM_CLASSES
from src.model.hybrid_model import HybridCNNLSTM
from src.explainability.gradcam import GradCAM1D


class TestGradCAM:
    """Tests for 1D Grad-CAM."""

    def setup_method(self):
        self.model = HybridCNNLSTM()
        self.model.eval()
        self.gradcam = GradCAM1D(self.model)

    def test_heatmap_shape(self):
        """Heatmap should match input signal length."""
        x = torch.randn(1, 1, BEAT_WINDOW)
        heatmap, pred, probs = self.gradcam.compute(x)
        assert heatmap.shape == (BEAT_WINDOW,)

    def test_heatmap_range(self):
        """Heatmap values should be in [0, 1]."""
        x = torch.randn(1, 1, BEAT_WINDOW)
        heatmap, _, _ = self.gradcam.compute(x)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0 + 1e-6

    def test_probs_sum_to_one(self):
        """Class probabilities should sum to 1."""
        x = torch.randn(1, 1, BEAT_WINDOW)
        _, _, probs = self.gradcam.compute(x)
        assert abs(probs.sum() - 1.0) < 1e-4

    def test_predicted_class_valid(self):
        """Predicted class should be a valid index."""
        x = torch.randn(1, 1, BEAT_WINDOW)
        _, pred, _ = self.gradcam.compute(x)
        assert 0 <= pred < NUM_CLASSES

    def test_specific_target_class(self):
        """Should compute heatmap for a specified target class."""
        x = torch.randn(1, 1, BEAT_WINDOW)
        heatmap, _, _ = self.gradcam.compute(x, target_class=0)
        assert heatmap.shape == (BEAT_WINDOW,)

    def test_batch_computation(self):
        """Batch computation should return list of results."""
        x = torch.randn(3, 1, BEAT_WINDOW)
        results = self.gradcam.compute_batch(x)
        assert len(results) == 3
        for hm, pred, probs in results:
            assert hm.shape == (BEAT_WINDOW,)

    def test_cleanup(self):
        """Hooks should be removable without error."""
        self.gradcam.remove_hooks()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
