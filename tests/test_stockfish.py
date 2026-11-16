"""Unit tests for chess/stockfish_engine.py."""

import pytest
import chess
from chess.stockfish_engine import StockfishEngine, get_stockfish_path


def test_stockfish_path_detection():
    """Verify that Stockfish executable path is detected."""
    path = get_stockfish_path()
    assert path is not None
    assert "stockfish" in path.lower()


def test_stockfish_lifecycle():
    """Test engine start, availability check, and safe shutdown."""
    engine = StockfishEngine(default_depth=10, default_time_limit=0.2)
    assert engine.is_available() is True

    started = engine.start()
    assert started is True

    # Closing engine
    engine.close()
    assert engine._engine is None


def test_stockfish_context_manager():
    """Test using StockfishEngine as context manager."""
    with StockfishEngine(default_depth=8, default_time_limit=0.2) as engine:
        assert engine._engine is not None
    assert engine._engine is None


def test_stockfish_startpos_evaluation():
    """Test evaluation of the starting chess position."""
    with StockfishEngine(default_depth=10, default_time_limit=0.3) as engine:
        board = chess.Board()
        eval_result = engine.evaluate_position(board)

        assert eval_result is not None
        assert "score_cp" in eval_result
        assert "score_formatted" in eval_result
        assert "best_move_san" in eval_result
        assert eval_result["best_move"] is not None
        assert eval_result["best_move"] in board.legal_moves
        assert eval_result["depth"] >= 8
        assert eval_result["nodes"] > 0
        assert len(eval_result["pv"]) > 0


def test_stockfish_tactical_mate_in_one():
    """Test evaluation on a mate-in-1 position."""
    # Scholar's Mate setup: Qxf7# is mate
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    with StockfishEngine(default_depth=12, default_time_limit=0.3) as engine:
        eval_result = engine.evaluate_position(board)

        assert eval_result["best_move_san"] == "Qxf7#"
        assert eval_result["mate_score"] == 1 or eval_result["score_cp"] >= 9000
        assert "M1" in eval_result["score_formatted"] or "+10" in eval_result["score_formatted"]


def test_stockfish_fallback_invalid_path():
    """Test graceful fallback when engine path is invalid."""
    engine = StockfishEngine(engine_path="non_existent_engine_file.exe")
    board = chess.Board()
    eval_result = engine.evaluate_position(board)

    assert eval_result["score_formatted"] == "0.00"
    assert eval_result["best_move_san"] == "-"
    assert eval_result["depth"] == 0
