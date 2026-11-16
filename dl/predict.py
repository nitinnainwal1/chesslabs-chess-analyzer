"""ChessLabs - Deep Learning Inference Engine.

Loads the PyTorch ChessCNN model checkpoint and performs real-time positional evaluation,
win probability calculation, and board assessment for any chess board.
"""

import os
from typing import Dict, Any, Optional
import chess
import torch
import numpy as np

from dl.dataset import board_to_tensor
from dl.model import ChessCNN


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "chess_position_model.pth",
)


class DLPredictor:
    """Inference engine for PyTorch Deep Learning board evaluation."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.model: Optional[ChessCNN] = None
        self.metrics: Optional[Dict[str, Any]] = None
        self.load_model()

    def is_model_loaded(self) -> bool:
        """Check if PyTorch weights are loaded and ready."""
        return self.model is not None

    def load_model(self) -> bool:
        """Load PyTorch checkpoint from disk."""
        if os.path.isfile(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=torch.device("cpu"), weights_only=True)
                self.model = ChessCNN()
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()
                self.metrics = checkpoint.get("metrics")
                return True
            except Exception:
                self.model = None
                return False
        return False

    def get_model_metrics(self) -> Optional[Dict[str, Any]]:
        """Return training metrics of the loaded DL model."""
        return self.metrics

    def predict_position(self, board: chess.Board) -> Dict[str, Any]:
        """Evaluate a chess board position using the Deep Learning CNN.

        Args:
            board: chess.Board position.

        Returns:
            Dictionary with predicted centipawns, win probabilities, and positional assessment.
        """
        if not self.is_model_loaded() or self.model is None:
            return {
                "model_available": False,
                "predicted_eval_cp": 0.0,
                "eval_formatted": "0.00",
                "white_win_pct": 50.0,
                "black_win_pct": 50.0,
                "assessment": "Model Not Loaded",
                "advantage_color": "#888888",
            }

        # 1. Convert board to tensor (1, 12, 8, 8)
        arr = board_to_tensor(board)
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

        # 2. Run model forward pass
        eval_cp = self.model.predict_centipawns(tensor)
        white_win_pct, black_win_pct = self.model.predict_win_probability(tensor)

        # 3. Assess positional standing
        eval_pawn = eval_cp / 100.0
        eval_formatted = f"+{eval_pawn:.2f}" if eval_cp >= 0 else f"{eval_pawn:.2f}"

        if eval_cp > 250:
            assessment = "Decisive White Advantage"
            adv_color = "#4caf50"
        elif eval_cp > 80:
            assessment = "Moderate White Advantage"
            adv_color = "#8bc34a"
        elif eval_cp > -80:
            assessment = "Balanced / Equal Position"
            adv_color = "#d4af37"
        elif eval_cp > -250:
            assessment = "Moderate Black Advantage"
            adv_color = "#03a9f4"
        else:
            assessment = "Decisive Black Advantage"
            adv_color = "#2196f3"

        return {
            "model_available": True,
            "predicted_eval_cp": eval_cp,
            "eval_formatted": eval_formatted,
            "white_win_pct": white_win_pct,
            "black_win_pct": black_win_pct,
            "assessment": assessment,
            "advantage_color": adv_color,
        }
