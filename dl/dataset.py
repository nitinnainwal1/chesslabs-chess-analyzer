"""ChessLabs - Deep Learning Tensor Encoding & Dataset.

Converts chess board states into 8x8x12 binary tensors representing piece locations
for convolutional neural networks in PyTorch.
"""

from typing import List, Tuple, Union, Optional
import chess
import numpy as np
import torch
from torch.utils.data import Dataset


# 12 piece channels: 6 White piece types (0-5), 6 Black piece types (6-11)
PIECE_TO_CHANNEL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Encode a python-chess Board into an (12, 8, 8) binary float32 numpy array.

    Channels:
        0-5: White Pawn, Knight, Bishop, Rook, Queen, King
        6-11: Black Pawn, Knight, Bishop, Rook, Queen, King

    Array Dimensions: (Channels=12, Ranks=8, Files=8)
    Rank 0 = Rank 1 (White's back rank), Rank 7 = Rank 8 (Black's back rank)
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        base_channel = PIECE_TO_CHANNEL[piece.piece_type]
        channel = base_channel if piece.color == chess.WHITE else base_channel + 6
        tensor[channel, rank, file] = 1.0

    return tensor


def fen_to_tensor(fen: str) -> np.ndarray:
    """Convert a FEN string directly into an (12, 8, 8) numpy tensor."""
    board = chess.Board(fen)
    return board_to_tensor(board)


class ChessPositionDataset(Dataset):
    """PyTorch Dataset of chess positions and corresponding evaluation targets."""

    def __init__(
        self,
        tensors: Union[np.ndarray, List[np.ndarray]],
        targets: Union[np.ndarray, List[float]],
    ) -> None:
        """Initialize dataset with tensors and evaluation targets.

        Args:
            tensors: Array of shape (N, 12, 8, 8).
            targets: Array of scalar evaluation targets (e.g. normalized centipawns or win-rate).
        """
        if isinstance(tensors, list):
            self.tensors = torch.tensor(np.array(tensors), dtype=torch.float32)
        elif isinstance(tensors, np.ndarray):
            self.tensors = torch.tensor(tensors, dtype=torch.float32)
        else:
            self.tensors = tensors

        if isinstance(targets, list):
            self.targets = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(1)
        elif isinstance(targets, np.ndarray):
            self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        else:
            self.targets = targets

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tensors[idx], self.targets[idx]
