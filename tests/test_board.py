"""Unit tests for chess/board.py."""

import pytest
import chess
import chess.svg
from chess.board import (
    generate_board_svg,
    get_board_from_fen,
    LIGHT_SQUARE_COLOR,
    DARK_SQUARE_COLOR,
)


def test_generate_board_svg_startpos():
    """Test generating SVG for initial chess position."""
    board = chess.Board()
    svg = generate_board_svg(board)
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    assert LIGHT_SQUARE_COLOR in svg
    assert DARK_SQUARE_COLOR in svg


def test_generate_board_svg_with_last_move():
    """Test SVG generation with highlighted last move."""
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    board.push(move)
    svg = generate_board_svg(board, last_move=move)
    assert svg.startswith("<svg")
    assert "lastmove" in svg.lower() or "class" in svg


def test_generate_board_svg_check():
    """Test SVG generation when king is in check."""
    # Scholar's mate final position (Checkmate/Check)
    board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    assert board.is_check()
    svg = generate_board_svg(board)
    assert "<svg" in svg
    assert "check" in svg.lower()


def test_generate_board_svg_flipped():
    """Test SVG generation with black orientation."""
    board = chess.Board()
    svg_white = generate_board_svg(board, flipped=False)
    svg_black = generate_board_svg(board, flipped=True)
    assert svg_white != svg_black


def test_generate_board_svg_with_arrows():
    """Test SVG generation with engine recommendation arrows."""
    board = chess.Board()
    arrow = chess.svg.Arrow(chess.E2, chess.E4, color="rgba(212, 175, 55, 0.8)")
    svg = generate_board_svg(board, arrows=[arrow])
    assert "<svg" in svg
    assert "marker" in svg or "line" in svg or "path" in svg


def test_get_board_from_fen_valid():
    """Test creating Board from valid FEN."""
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    board = get_board_from_fen(fen)
    assert board.fen() == fen
    assert board.turn == chess.BLACK
    assert board.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)


def test_get_board_from_fen_invalid_fallback():
    """Test fallback to standard board on invalid FEN."""
    board = get_board_from_fen("invalid-fen-string")
    assert board.fen() == chess.STARTING_FEN


def test_generate_board_svg_with_classification_badge():
    """Test SVG generation includes on-piece classification badge."""
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    board.push(move)
    svg = generate_board_svg(board, last_move=move, classification="Best")
    assert "move-classification-badge" in svg
    assert "★" in svg

    svg_blunder = generate_board_svg(board, last_move=move, classification="Blunder")
    assert "move-classification-badge" in svg_blunder
    assert "??" in svg_blunder

