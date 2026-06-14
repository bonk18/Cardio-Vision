"""
CardioVision — Model Unit Tests
Tests for CNN, LSTM, Hybrid model, and dataset.
"""

import numpy as np
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BEAT_WINDOW, NUM_CLASSES
from src.model.cnn_module import CNN1DEncoder
from src.model.lstm_module import LSTMEncoder
from src.model.hybrid_model import HybridCNNLSTM
from src.model.trainer import ECGDataset


class TestCNN:
    """Tests for the 1D CNN Encoder."""

    def test_output_shape(self):
        """CNN should produce expected output dimensions."""
        model = CNN1DEncoder()
        x = torch.randn(4, 1, BEAT_WINDOW)  # batch=4
        out = model(x)
        assert out.dim() == 3
        assert out.shape[0] == 4
        assert out.shape[1] == 128  # final channel count

    def test_get_last_conv(self):
        """Should return a Conv1d layer."""
        model = CNN1DEncoder()
        layer = model.get_last_conv_layer()
        assert isinstance(layer, torch.nn.Conv1d)

    def test_single_sample(self):
        """Should work with batch_size=1."""
        model = CNN1DEncoder()
        x = torch.randn(1, 1, BEAT_WINDOW)
        out = model(x)
        assert out.shape[0] == 1


class TestLSTM:
    """Tests for the LSTM Encoder."""

    def test_output_shape(self):
        """BiLSTM should output hidden_size*2 features."""
        model = LSTMEncoder(input_size=128, hidden_size=64, bidirectional=True)
        x = torch.randn(4, 45, 128)  # (batch, seq, features)
        out = model(x)
        assert out.shape == (4, 128)  # 64*2 for bidirectional

    def test_unidirectional(self):
        """Unidirectional LSTM should output hidden_size features."""
        model = LSTMEncoder(input_size=128, hidden_size=64, bidirectional=False)
        x = torch.randn(4, 45, 128)
        out = model(x)
        assert out.shape == (4, 64)


class TestHybridModel:
    """Tests for the full Hybrid CNN-LSTM model."""

    def setup_method(self):
        self.model = HybridCNNLSTM()

    def test_output_shape(self):
        """Model should output (batch, NUM_CLASSES) logits."""
        x = torch.randn(8, 1, BEAT_WINDOW)
        out = self.model(x)
        assert out.shape == (8, NUM_CLASSES)

    def test_predict_proba(self):
        """Softmax probabilities should sum to 1."""
        x = torch.randn(4, 1, BEAT_WINDOW)
        probs = self.model.predict_proba(x)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_gradient_flow(self):
        """Gradients should flow through the full model."""
        x = torch.randn(2, 1, BEAT_WINDOW, requires_grad=True)
        out = self.model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.any(x.grad != 0)

    def test_cuda_if_available(self):
        """Model should move to CUDA if available."""
        if torch.cuda.is_available():
            model = HybridCNNLSTM().cuda()
            x = torch.randn(2, 1, BEAT_WINDOW).cuda()
            out = model(x)
            assert out.is_cuda


class TestDataset:
    """Tests for ECGDataset."""

    def test_dataset_shapes(self):
        """Dataset should return correct tensor shapes."""
        X = np.random.randn(100, BEAT_WINDOW).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, 100).astype(np.int64)
        ds = ECGDataset(X, y)
        assert len(ds) == 100
        sample, label = ds[0]
        assert sample.shape == (1, BEAT_WINDOW)
        assert label.dim() == 0  # scalar


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
