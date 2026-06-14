"""
CardioVision — LSTM Encoder
Temporal dependency modeling for rhythmic irregularity detection.
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_BIDIRECTIONAL, LSTM_DROPOUT


class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for temporal sequence modeling.

    Takes the spatially-encoded features from the CNN and models
    temporal dependencies across the beat's time steps. This captures
    rhythmic patterns like inter-beat intervals and temporal
    irregularities that indicate arrhythmias.

    Input:  (batch, seq_len, features) — CNN feature maps permuted.
    Output: (batch, hidden_dim) — final temporal representation.
    """

    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        bidirectional: bool = LSTM_BIDIRECTIONAL,
        dropout: float = LSTM_DROPOUT,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)
        self.output_size = hidden_size * self.num_directions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size) — feature sequences.

        Returns
        -------
        h_final : (batch, hidden_size * num_directions)
            Concatenation of forward and backward final hidden states.
        """
        # LSTM forward pass
        output, (h_n, _) = self.lstm(x)

        # Concat final hidden states from both directions
        if self.bidirectional:
            # h_n shape: (num_layers * 2, batch, hidden_size)
            h_forward = h_n[-2]  # Last forward layer
            h_backward = h_n[-1]  # Last backward layer
            h_final = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_final = h_n[-1]

        h_final = self.layer_norm(h_final)
        return h_final
