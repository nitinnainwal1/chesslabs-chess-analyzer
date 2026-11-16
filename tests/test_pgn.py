"""Unit tests for chess/pgn_parser.py."""

import pytest
from chess.pgn_parser import (
    parse_pgn,
    validate_pgn,
    extract_game_metadata,
    extract_moves_and_positions,
)

# Sample valid PGN
VALID_PGN_OPERA = """[Event "Paris Opera House"]
[Site "Paris FRA"]
[Date "1858.10.21"]
[Round "?"]
[White "Paul Morphy"]
[Black "Duke Karl / Count Isouard"]
[Result "1-0"]
[ECO "C41"]
[Opening "Philidor Defense"]

1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6 7. Qb3 Qe7 8. Nc3 c6 9. Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7 12. O-O-O Rd8 13. Rxd7 Rxd7 14. Rd1 Qe6 15. Bxd7+ Nxd7 16. Qb8+ Nxb8 17. Rd8# 1-0
"""

# Invalid PGN with illegal move
INVALID_PGN_ILLEGAL_MOVE = """[Event "Invalid Game"]
[White "Player A"]
[Black "Player B"]
[Result "*"]

1. e4 e5 2. Ke2 Ke7 3. Ke8# *
"""

# PGN with no moves
PGN_NO_MOVES = """[Event "Empty Moves Game"]
[White "Player A"]
[Black "Player B"]
[Result "*"]
"""


def test_parse_valid_pgn():
    """Test parsing a known valid historical game."""
    game = parse_pgn(VALID_PGN_OPERA)
    assert game is not None
    assert game.headers["White"] == "Paul Morphy"
    assert game.headers["Result"] == "1-0"


def test_parse_empty_pgn():
    """Test parsing empty or whitespace string."""
    assert parse_pgn("") is None
    assert parse_pgn("   \n\t  ") is None


def test_validate_valid_pgn():
    """Test validation of valid PGN."""
    is_valid, msg, meta = validate_pgn(VALID_PGN_OPERA)
    assert is_valid is True
    assert meta is not None
    assert meta["white"] == "Paul Morphy"
    assert meta["black"] == "Duke Karl / Count Isouard"
    assert meta["total_plies"] == 33  # 17 White moves, 16 Black moves
    assert meta["eco"] == "C41"


def test_validate_empty_pgn():
    """Test validation returns False for empty strings."""
    is_valid, msg, meta = validate_pgn("")
    assert is_valid is False
    assert "empty" in msg.lower()
    assert meta is None


def test_validate_no_moves_pgn():
    """Test validation fails when PGN has headers but no moves."""
    is_valid, msg, meta = validate_pgn(PGN_NO_MOVES)
    assert is_valid is False
    assert "valid moves" in msg.lower()


def test_validate_illegal_move_pgn():
    """Test validation detects illegal moves in game."""
    is_valid, msg, meta = validate_pgn(INVALID_PGN_ILLEGAL_MOVE)
    assert is_valid is False


def test_extract_game_metadata_defaults():
    """Test fallback defaults when headers are missing."""
    minimal_pgn = "1. e4 e5 2. Nf3 Nc6 1-0"
    game = parse_pgn(minimal_pgn)
    assert game is not None
    meta = extract_game_metadata(game)
    assert meta["white"] == "White Player"
    assert meta["black"] == "Black Player"
    assert meta["event"] == "Casual Game"


def test_extract_moves_and_positions():
    """Test extracting chronological position list (FENs and SANs)."""
    short_pgn = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
    game = parse_pgn(short_pgn)
    assert game is not None
    positions = extract_moves_and_positions(game)

    # Total 7 plies + 1 starting position = 8 entries
    assert len(positions) == 8
    assert positions[0]["ply"] == 0
    assert positions[0]["fen"].startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")

    # Ply 1: 1. e4 (White)
    assert positions[1]["ply"] == 1
    assert positions[1]["san"] == "e4"
    assert positions[1]["is_white"] is True

    # Ply 7: 4. Qxf7# (Checkmate)
    assert positions[7]["ply"] == 7
    assert positions[7]["san"] == "Qxf7#"
    assert positions[7]["capture"] is True
    assert positions[7]["check"] is True
