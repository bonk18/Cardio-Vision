import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import NUM_CLASSES, FC_HIDDEN, FC_DROPOUT
from src.model.cnn_module import CNN1DEncoder
from src.model.lstm_module import LSTMEncoder

class SelfAttention1D(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** 0.5

    def forward(self, x):
        # x: (B, seq_len, hidden_size)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # (B, seq_len, hidden_size) x (B, hidden_size, seq_len) -> (B, seq_len, seq_len)
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # (B, seq_len, seq_len) x (B, seq_len, hidden_size) -> (B, seq_len, hidden_size)
        attended_x = torch.bmm(attention_weights, V)
        return attended_x

class AttentionHybridModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.cnn = CNN1DEncoder()
        
        # Attention applied on the permuted CNN features
        self.attention = SelfAttention1D(self.cnn.out_channels)
        
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
        # x: (B, 1, 360)
        cnn_features = self.cnn(x)  # (B, C, L)
        
        # Permute for attention and LSTM
        lstm_input = cnn_features.permute(0, 2, 1)  # (B, L, C)
        
        # Apply self-attention
        attended_features = self.attention(lstm_input)
        
        # LSTM
        temporal_features = self.lstm(attended_features)
        
        # Classifier
        logits = self.classifier(temporal_features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def get_cnn_encoder(self):
        return self.cnn
