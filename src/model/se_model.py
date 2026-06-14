import torch
import torch.nn as nn
from typing import List

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import CNN_CHANNELS, CNN_KERNEL_SIZES, CNN_POOL_SIZE, CNN_DROPOUT, NUM_CLASSES, FC_HIDDEN, FC_DROPOUT
from src.model.lstm_module import LSTMEncoder

class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SECNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int = CNN_POOL_SIZE,
        dropout: float = CNN_DROPOUT,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock1D(out_channels)
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)  # Apply Squeeze-and-Excitation
        x = self.pool(x)
        x = self.dropout(x)
        return x

class SECNN1DEncoder(nn.Module):
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

        layers = []
        for i in range(len(kernel_sizes)):
            layers.append(
                SECNNBlock(channels[i], channels[i + 1], kernel_sizes[i], pool_size, dropout)
            )

        self.encoder = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def get_last_conv_layer(self):
        return self.encoder[-1].conv

class SEHybridModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.cnn = SECNN1DEncoder()
        self.lstm = LSTMEncoder(input_size=self.cnn.out_channels)
        
        lstm_out = self.lstm.output_size
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, FC_HIDDEN),
            nn.BatchNorm1d(FC_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(FC_DROPOUT),
            nn.Linear(FC_HIDDEN, num_classes),
        )

    def forward(self, x):
        cnn_features = self.cnn(x)
        lstm_input = cnn_features.permute(0, 2, 1)
        temporal_features = self.lstm(lstm_input)
        logits = self.classifier(temporal_features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def get_cnn_encoder(self):
        return self.cnn
