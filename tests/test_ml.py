"""Unit tests for ml/features.py and ml/predict.py."""

import os
import pytest
import chess
from ml.features import (
    extract_features,
    calculate_material,
    calculate_center_control,
    calculate_king_safety,
    FEATURE_COLUMNS,
)
from ml.predict import MLMovePredictor


def test_calculate_material_starting_pos():
    """Test material balance on initial board position."""
    board = chess.Board()
    mat = calculate_material(board)
    assert mat["material_balance"] == 0
    assert mat["white_material"] == mat["black_material"]
    assert mat["total_material"] > 0


def test_calculate_center_control():
    """Test center control computation."""
    board = chess.Board()
    ctrl = calculate_center_control(board)
    assert "white_center_control" in ctrl
    assert "black_center_control" in ctrl
    assert ctrl["center_control_diff"] == 0


def test_extract_features_structure():
    """Test that extracted feature dict contains all expected numeric columns."""
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    feats = extract_features(board, move, eval_before_cp=20)

    for col in FEATURE_COLUMNS:
        assert col in feats, f"Missing feature: {col}"
        assert isinstance(feats[col], (int, float)), f"Feature {col} is not numeric"


def test_ml_predictor_inference():
    """Test ML predictor loads saved model and produces valid predictions."""
    predictor = MLMovePredictor()
    assert predictor.is_model_loaded() is True

    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    pred = predictor.predict_move(board, move, eval_before_cp=20)

    assert pred["model_available"] is True
    assert 0.0 <= pred["blunder_probability"] <= 100.0
    assert 0.0 <= pred["confidence"] <= 100.0
    assert "risk_level" in pred
    assert "features" in pred
