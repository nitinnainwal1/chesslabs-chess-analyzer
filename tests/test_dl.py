"""ChessLabs - Deep Learning Unit Tests.

Tests for 8x8x12 tensor board encoding, PyTorch CNN architecture,
and deep learning positional evaluation inference.
"""

import os
import pytest
import chess
import numpy as np
import torch

from dl.dataset import board_to_tensor, fen_to_tensor, ChessPositionDataset
from dl.model import ChessCNN
from dl.predict import DLPredictor


def test_board_to_tensor_shape_and_values():
    """Verify board_to_tensor produces correct (12, 8, 8) shape and piece placements."""
    board = chess.Board()
    tensor = board_to_tensor(board)

    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (12, 8, 8)
    assert tensor.dtype == np.float32

    # In starting position, White Pawns on Rank 2 (index 1) -> Channel 0
    assert np.sum(tensor[0, 1, :]) == 8.0
    # Black Pawns on Rank 7 (index 6) -> Channel 6
    assert np.sum(tensor[6, 6, :]) == 8.0
    # Total piece count = 32
    assert np.sum(tensor) == 32.0


def test_fen_to_tensor_consistency():
    """Verify fen_to_tensor produces identical representation to board_to_tensor."""
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    board = chess.Board(fen)

    t1 = board_to_tensor(board)
    t2 = fen_to_tensor(fen)

    np.testing.assert_array_equal(t1, t2)


def test_chess_cnn_forward_pass():
    """Verify ChessCNN forward pass output dimensions and value ranges."""
    model = ChessCNN()
    model.eval()

    # Batch of 4 random or starting boards
    dummy_input = torch.zeros((4, 12, 8, 8), dtype=torch.float32)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (4, 1)
    # Output should be bounded in [-1.0, 1.0] by tanh
    assert torch.all(output >= -1.0) and torch.all(output <= 1.0)


def test_dl_predictor_inference():
    """Test DLPredictor loads trained model and outputs valid evaluations."""
    predictor = DLPredictor()
    
    # Even if model file isn't found, should handle gracefully
    board = chess.Board()
    result = predictor.predict_position(board)

    assert isinstance(result, dict)
    assert "model_available" in result
    assert "predicted_eval_cp" in result
    assert "white_win_pct" in result
    assert "black_win_pct" in result
    assert "assessment" in result

    if result["model_available"]:
        assert 0.0 <= result["white_win_pct"] <= 100.0
        assert 0.0 <= result["black_win_pct"] <= 100.0
        assert round(result["white_win_pct"] + result["black_win_pct"], 1) == 100.0
