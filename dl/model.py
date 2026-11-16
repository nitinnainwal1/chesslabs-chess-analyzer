"""ChessLabs - Deep Learning PyTorch Model Architecture.

A Convolutional Neural Network (CNN) that takes an (12, 8, 8) board representation
and predicts the positional evaluation and win probability.
"""

from typing import Dict, Any, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessCNN(nn.Module):
    """Deep Convolutional Neural Network for Board Position Evaluation.
    
    Processes 12 board feature channels across the 8x8 chessboard grid.
    Uses convolutional filters to capture piece interactions, pawn structures,
    and king safety patterns.
    """

    def __init__(self) -> None:
        super().__init__()
        # Convolutional Feature Extractor
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Dense Evaluation Layers
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(128, 64)
        
        # Output evaluation head (scalar in range roughly -1.0 to 1.0, representing cp / 1000)
        self.eval_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (Batch, 12, 8, 8).
            
        Returns:
            Output tensor of shape (Batch, 1) with normalized evaluation scores.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        # We use tanh to bound the output to [-1.0, +1.0] representing +/- 1000 centipawns
        out = torch.tanh(self.eval_head(x))
        return out

    def predict_centipawns(self, x: torch.Tensor) -> float:
        """Predict centipawn evaluation from moving player / White perspective.
        
        Scaled so that +/- 1.0 maps to +/- 1000 centipawns.
        """
        self.eval()
        with torch.no_grad():
            score_norm = self.forward(x).item()
            cp = score_norm * 1000.0
            return round(cp, 1)

    def predict_win_probability(self, x: torch.Tensor) -> Tuple[float, float]:
        """Predict White and Black win probability percentages using logistic sigmoid.
        
        Returns:
            (white_win_pct, black_win_pct) in range [0.0, 100.0]
        """
        cp = self.predict_centipawns(x)
        # Standard chess win probability formula from centipawns: P(win) = 1 / (1 + 10^(-cp / 400))
        white_prob = 1.0 / (1.0 + math.pow(10.0, -cp / 400.0))
        white_pct = round(white_prob * 100.0, 1)
        black_pct = round((1.0 - white_prob) * 100.0, 1)
        return white_pct, black_pct
