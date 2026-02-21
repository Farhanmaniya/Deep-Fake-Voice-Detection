"""
backend/core/deepfake_cnn.py

DeepfakeCNN architecture — must match the notebook exactly so
torch.load + load_state_dict works without key mismatches.

Input : (batch, 1, 40, 174)  — 1-channel MFCC map
Output: (batch, 2)           — raw logits [real_logit, fake_logit]
"""

import torch
import torch.nn as nn


class DeepfakeCNN(nn.Module):
    """
    CNN for deepfake voice detection trained on MFCC features.

    Architecture:
        Input: (batch, 1, 40, 174)
          → Conv2d(1→32)  + BN + ReLU + MaxPool2d  →  (32, 20, 87)
          → Conv2d(32→64) + BN + ReLU + MaxPool2d  →  (64, 10, 43)
          → Conv2d(64→128)+ BN + ReLU + MaxPool2d  →  (128, 5, 21)
          → Flatten → 13 440
          → FC(13440→256) + Dropout(0.4) + ReLU
          → FC(256→64)    + Dropout(0.4) + ReLU
          → FC(64→2)      → raw logits
    """

    def __init__(self):
        super(DeepfakeCNN, self).__init__()

        # Block 1: Conv → BN → ReLU → MaxPool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)        # 40×174 → 20×87

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)        # 20×87  → 10×43

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)        # 10×43  → 5×21

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.4)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 5 * 21, 256)   # 13 440 → 256
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)                # 2 output neurons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)               # flatten → (batch, 13440)
        x = self.drop(self.relu(self.fc1(x)))
        x = self.drop(self.relu(self.fc2(x)))
        return self.fc3(x)                       # raw logits (batch, 2)
