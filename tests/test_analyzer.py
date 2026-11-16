"""Unit tests for chess/move_analyzer.py and chess/game_analyzer.py."""

import os
import pytest
import chess
from chess.move_analyzer import (
    calculate_centipawn_loss,
    classify_move,
    calculate_move_accuracy,
    is_material_sacrifice,
    CLASS_BRILLIANT,
    CLASS_GREAT,
    CLASS_BEST,
    CLASS_EXCELLENT,
    CLASS_GOOD,
    CLASS_MISS,
    CLASS_INACCURACY,
    CLASS_MISTAKE,
    CLASS_BLUNDER,
)
from chess.game_analyzer import (
    analyze_full_game,
    create_evaluation_chart,
    create_move_breakdown_chart,
)
from chess.stockfish_engine import StockfishEngine
from chess.pgn_parser import parse_pgn


def test_centipawn_loss_perspective():
    """Test that centipawn loss correctly measures drops from player POV."""
    # Player had +100 cp advantage and dropped to +20 cp -> Loss = 80 cp
    loss = calculate_centipawn_loss(100, 20)
    assert loss == 80

    # Player improved position from +20 to +50 -> Loss = 0
    loss_improvement = calculate_centipawn_loss(20, 50)
    assert loss_improvement == 0

    # Mate lost -> severe loss (600 cp)
    loss_mate = calculate_centipawn_loss(10000, 200, is_mate_before=True, is_mate_after=False)
    assert loss_mate == 600

    # Mate delivered or kept -> 0 loss
    loss_mate_kept = calculate_centipawn_loss(10000, 10000, is_mate_before=True, is_mate_after=True)
    assert loss_mate_kept == 0


def test_move_classification_all_tiers():
    """Verify classification across all standard chess review categories."""
    # 1. Brilliant: Best move with material sacrifice leading to winning advantage
    assert classify_move(is_best_move=True, centipawn_loss=0, is_sacrifice=True, eval_after_turn=250) == CLASS_BRILLIANT

    # 2. Great Move: High-impact comeback move in tough position
    assert classify_move(is_best_move=True, centipawn_loss=0, eval_before_turn=-150, eval_after_turn=10) == CLASS_GREAT

    # 3. Best Move
    assert classify_move(is_best_move=True, centipawn_loss=0) == CLASS_BEST
    assert classify_move(is_best_move=False, centipawn_loss=8) == CLASS_BEST

    # 4. Excellent Move
    assert classify_move(is_best_move=False, centipawn_loss=20) == CLASS_EXCELLENT

    # 5. Good Move
    assert classify_move(is_best_move=False, centipawn_loss=40) == CLASS_GOOD

    # 6. Miss: Had winning position (+300) but failed tactic and dropped to +20 (loss = 280)
    assert classify_move(is_best_move=False, centipawn_loss=280, eval_before_turn=300, eval_after_turn=20) == CLASS_MISS

    # 7. Inaccuracy
    assert classify_move(is_best_move=False, centipawn_loss=90) == CLASS_INACCURACY

    # 8. Mistake
    assert classify_move(is_best_move=False, centipawn_loss=180) == CLASS_MISTAKE

    # 9. Blunder
    assert classify_move(is_best_move=False, centipawn_loss=350) == CLASS_BLUNDER


def test_is_material_sacrifice():
    """Verify material sacrifice detection logic."""
    # 1. Greek Gift Sacrifice: Bxh7+
    board = chess.Board("r1bq1rk1/ppp2ppp/2np4/4p3/2B1P3/3P1N2/PPP2PPP/R1BQK2R w KQ - 0 1")
    # Move knight or bishop to attacked square
    board_greek = chess.Board("r1bq1rk1/pppn1ppp/4p3/3p4/1bPP4/2N1PN2/PP1Q1PPP/R3KB1R w KQ - 0 1")
    move_sac = chess.Move.from_uci("f1h7")  # Bishop from f1 to h7 (not legal here, but test setup)
    
    # Test real Greek Gift position
    b_gift = chess.Board("r1bq1rk1/ppp2ppp/2n1pn2/3p4/1bPP4/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 1")
    # Real sacrifice: Queen takes attacked piece
    b_sac = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    # Normal pawn push is NOT a sacrifice
    move_pawn = chess.Move.from_uci("d2d4")
    assert is_material_sacrifice(b_sac, move_pawn) is False


def test_accuracy_calculation():
    """Test accuracy calculations."""
    assert calculate_move_accuracy(0) == 100.0
    assert 80.0 <= calculate_move_accuracy(40) <= 90.0
    assert calculate_move_accuracy(500) < 20.0
    assert 0.0 <= calculate_move_accuracy(1000) <= 100.0


def test_full_game_analysis_scholars_mate():
    """Test full game analysis on Scholar's Mate sample game."""
    pgn_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample", "scholars_mate.pgn")
    with open(pgn_path, "r", encoding="utf-8") as f:
        game = parse_pgn(f.read())
    assert game is not None

    with StockfishEngine(default_depth=10, default_time_limit=0.2) as engine:
        results = analyze_full_game(game, engine, depth=10, time_limit=0.2)

        assert results is not None
        assert results["total_plies"] == 7
        assert len(results["positions"]) == 8  # Start + 7 plies
        assert results["white_accuracy"] > 75.0  # White executed clean Scholar's Mate attack
        # Checkmate move Qxf7# must be Best move, not blunder
        last_pos = results["positions"][-1]
        assert last_pos["san"] == "Qxf7#"
        assert last_pos["classification"] in (CLASS_BEST, CLASS_BRILLIANT, CLASS_GREAT)
        assert results["black_counts"][CLASS_BLUNDER] >= 1  # Black blundered with 3... Nf6 allowing Qxf7#
        assert len(results["critical_moments"]) >= 1


def test_create_evaluation_chart_and_breakdown():
    """Verify that Plotly figures are generated with expected traces and properties."""
    sample_positions = [
        {"ply": 0, "move_number": 0, "is_white": True, "san": "", "eval_white": 20, "eval_formatted": "+0.20", "classification": CLASS_BEST, "cpl": 0},
        {"ply": 1, "move_number": 1, "is_white": True, "san": "e4", "eval_white": 30, "eval_formatted": "+0.30", "classification": CLASS_BEST, "cpl": 0},
        {"ply": 2, "move_number": 1, "is_white": False, "san": "e5", "eval_white": 25, "eval_formatted": "+0.25", "classification": CLASS_GOOD, "cpl": 15},
        {"ply": 3, "move_number": 2, "is_white": True, "san": "Qh5", "eval_white": -20, "eval_formatted": "-0.20", "classification": CLASS_INACCURACY, "cpl": 50},
        {"ply": 4, "move_number": 2, "is_white": False, "san": "Nf6", "eval_white": 900, "eval_formatted": "+9.00", "classification": CLASS_BLUNDER, "cpl": 900},
    ]

    fig_eval = create_evaluation_chart(sample_positions, current_ply=2)
    assert fig_eval is not None
    assert len(fig_eval.data) >= 2  # Evaluation curve + blunder/cursor markers

    fig_breakdown = create_move_breakdown_chart(
        white_counts={CLASS_BEST: 1, CLASS_GOOD: 0, CLASS_INACCURACY: 1, CLASS_MISTAKE: 0, CLASS_BLUNDER: 0},
        black_counts={CLASS_BEST: 0, CLASS_GOOD: 1, CLASS_INACCURACY: 0, CLASS_MISTAKE: 0, CLASS_BLUNDER: 1},
    )
    assert fig_breakdown is not None
    assert len(fig_breakdown.data) == 2  # White bar + Black bar
