import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import NUM_CLASSES, FC_HIDDEN, FC_DROPOUT
from src.model.lstm_module import LSTMEncoder

class ResNet1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out

class ResNet1DEncoder(nn.Module):
    def __init__(self, channels=[1, 64, 128, 256]):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv1d(channels[0], channels[1], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = ResNet1DBlock(channels[1], channels[1], stride=1)
        self.layer2 = ResNet1DBlock(channels[1], channels[2], stride=2)
        self.layer3 = ResNet1DBlock(channels[2], channels[3], stride=2)
        
        self.out_channels = channels[-1]

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def get_last_conv_layer(self):
        # Return the last conv layer for Grad-CAM
        return self.layer3.conv2

class ResNetHybridModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.cnn = ResNet1DEncoder()
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
