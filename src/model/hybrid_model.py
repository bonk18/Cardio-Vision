"""
CardioVision — Hybrid CNN-LSTM Model
Fused architecture for cardiac arrhythmia classification.
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    NUM_CLASSES, CNN_CHANNELS, CNN_KERNEL_SIZES, CNN_POOL_SIZE, CNN_DROPOUT,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_BIDIRECTIONAL, LSTM_DROPOUT,
    FC_HIDDEN, FC_DROPOUT, BEAT_WINDOW,
)
from src.model.cnn_module import CNN1DEncoder
from src.model.lstm_module import LSTMEncoder


class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model for ECG arrhythmia classification.

    Architecture:
        1. 1D CNN Encoder — extracts spatial/morphological features
           from the ECG waveform shape (P, QRS, T waves).
        2. BiLSTM Encoder — models temporal dependencies across
           the feature sequence to detect rhythmic irregularities.
        3. Fully Connected Head — classifies into arrhythmia types.

    Input:  (batch, 1, BEAT_WINDOW)  e.g. (B, 1, 360)
    Output: (batch, NUM_CLASSES)     e.g. (B, 4)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        cnn_channels=None,
        cnn_kernels=None,
        cnn_pool=CNN_POOL_SIZE,
        cnn_dropout=CNN_DROPOUT,
        lstm_hidden=LSTM_HIDDEN_SIZE,
        lstm_layers=LSTM_NUM_LAYERS,
        lstm_bidir=LSTM_BIDIRECTIONAL,
        lstm_dropout=LSTM_DROPOUT,
        fc_hidden=FC_HIDDEN,
        fc_dropout=FC_DROPOUT,
    ):
        super().__init__()

        # 1) CNN Encoder for spatial features
        self.cnn = CNN1DEncoder(
            channels=cnn_channels or CNN_CHANNELS,
            kernel_sizes=cnn_kernels or CNN_KERNEL_SIZES,
            pool_size=cnn_pool,
            dropout=cnn_dropout,
        )

        # 2) LSTM Encoder for temporal dependencies
        self.lstm = LSTMEncoder(
            input_size=self.cnn.out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=lstm_bidir,
            dropout=lstm_dropout,
        )

        # 3) Classification Head
        lstm_out = self.lstm.output_size
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier/Kaiming initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full pipeline.

        Parameters
        ----------
        x : (batch, 1, seq_len) — single-channel ECG beat.

        Returns
        -------
        logits : (batch, num_classes) — raw class scores (pre-softmax).
        """
        # CNN: (B, 1, 360) → (B, 128, 45)
        cnn_features = self.cnn(x)

        # Permute for LSTM: (B, 128, 45) → (B, 45, 128)
        lstm_input = cnn_features.permute(0, 2, 1)

        # LSTM: (B, 45, 128) → (B, 128) [bidirectional concat]
        temporal_features = self.lstm(lstm_input)

        # Classifier: (B, 128) → (B, 4)
        logits = self.classifier(temporal_features)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def get_cnn_encoder(self) -> CNN1DEncoder:
        """Return the CNN sub-module (for Grad-CAM)."""
        return self.cnn


def build_model(device: torch.device = None) -> HybridCNNLSTM:
    """Factory function to create and move model to device."""
    from config import DEVICE
    device = device or DEVICE
    model = HybridCNNLSTM()
    model = model.to(device)
    return model
