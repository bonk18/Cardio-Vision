"""
CardioVision — 1D CNN Encoder
Spatial feature extraction from ECG beat morphology.
"""

import torch
import torch.nn as nn
from typing import List

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import CNN_CHANNELS, CNN_KERNEL_SIZES, CNN_POOL_SIZE, CNN_DROPOUT


class CNNBlock(nn.Module):
    """Single 1D convolutional block: Conv → BatchNorm → ReLU → MaxPool → Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int = CNN_POOL_SIZE,
        dropout: float = CNN_DROPOUT,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=kernel_size // 2,  # 'same' padding
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN1DEncoder(nn.Module):
    """
    Stack of 1D CNN blocks for spatial feature extraction.

    Input:  (batch, 1, seq_len)      e.g. (B, 1, 360)
    Output: (batch, C_out, seq_len') e.g. (B, 128, 45)

    Extracts morphological patterns (P-wave, QRS complex, T-wave)
    through progressively deeper convolutional layers.
    """

    def __init__(
        self,
        channels: List[int] = None,
        kernel_sizes: List[int] = None,
        pool_size: int = CNN_POOL_SIZE,
        dropout: float = CNN_DROPOUT,
    ):
        super().__init__()
        channels = channels or CNN_CHANNELS
        kernel_sizes = kernel_sizes or CNN_KERNEL_SIZES

        assert len(channels) - 1 == len(kernel_sizes), (
            f"Need {len(channels)-1} kernel sizes for {len(channels)} channel specs"
        )

        layers = []
        for i in range(len(kernel_sizes)):
            layers.append(
                CNNBlock(channels[i], channels[i + 1], kernel_sizes[i], pool_size, dropout)
            )

        self.encoder = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, 1, seq_len) — raw ECG beat segment.

        Returns
        -------
        features : (batch, out_channels, reduced_seq_len)
        """
        return self.encoder(x)

    def get_last_conv_layer(self) -> nn.Module:
        """Return the last Conv1d layer (for Grad-CAM hooks)."""
        # The last CNNBlock's first element is Conv1d
        last_block = self.encoder[-1]
        return last_block.block[0]  # Conv1d inside Sequential
