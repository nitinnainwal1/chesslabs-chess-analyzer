"""ChessLabs - PGN Parser & Validator.

Handles parsing, validating, and extracting metadata and move-by-move
positions from Portable Game Notation (PGN) data.
"""

import io
from typing import Any, Dict, List, Optional, Tuple
import chess
import chess.pgn


class PGNParseException(Exception):
    """Exception raised when PGN parsing fails."""
    pass


def parse_pgn(pgn_text: str) -> Optional[chess.pgn.Game]:
    """Parse a PGN string into a python-chess Game object.

    Args:
        pgn_text: String containing PGN data.

    Returns:
        chess.pgn.Game object if parsing succeeds and game has moves/headers,
        None if empty.

    Raises:
        PGNParseException: If the PGN is malformed or invalid.
    """
    if not pgn_text or not pgn_text.strip():
        return None

    try:
        pgn_io = io.StringIO(pgn_text.strip())
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            raise PGNParseException("Could not parse any valid chess game from the provided PGN.")

        # If parser encountered syntax or legality errors
        if hasattr(game, "errors") and game.errors:
            raise PGNParseException(f"PGN error: {'; '.join(str(e) for e in game.errors)}")

        return game
    except Exception as e:
        if isinstance(e, PGNParseException):
            raise
        raise PGNParseException(f"PGN syntax error: {str(e)}") from e


def validate_pgn(pgn_text: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Validate a PGN string and return validation status with game summary.

    Args:
        pgn_text: PGN string to validate.

    Returns:
        Tuple of (is_valid, message, metadata_or_none)
    """
    if not pgn_text or not pgn_text.strip():
        return False, "PGN input is empty.", None

    try:
        game = parse_pgn(pgn_text)
        if game is None:
            return False, "No game content found in PGN.", None

        # Check if the game has at least one move
        moves = list(game.mainline_moves())
        if not moves:
            return False, "PGN does not contain any valid moves.", None

        board = game.board()
        for move in moves:
            if move not in board.legal_moves:
                return False, f"Illegal move found in PGN: {move.uci()}", None
            board.push(move)

        metadata = extract_game_metadata(game)
        metadata["total_moves"] = (len(moves) + 1) // 2
        metadata["total_plies"] = len(moves)

        return True, f"Valid PGN loaded successfully ({len(moves)} plies).", metadata

    except PGNParseException as e:
        return False, str(e), None
    except Exception as e:
        return False, f"Unexpected validation error: {str(e)}", None


def extract_game_metadata(game: chess.pgn.Game) -> Dict[str, str]:
    """Extract standard header metadata from a parsed chess game.

    Args:
        game: Parsed chess.pgn.Game instance.

    Returns:
        Dictionary of header tags with clean fallback defaults.
    """
    headers = game.headers

    def clean_header(key: str, default: str) -> str:
        val = headers.get(key, default)
        return default if val in (None, "?", "", "????.??.??") else val

    return {
        "event": clean_header("Event", "Casual Game"),
        "site": clean_header("Site", "Unknown"),
        "date": clean_header("Date", "Unknown Date"),
        "round": clean_header("Round", "-"),
        "white": clean_header("White", "White Player"),
        "black": clean_header("Black", "Black Player"),
        "result": clean_header("Result", "*"),
        "white_elo": clean_header("WhiteElo", "-"),
        "black_elo": clean_header("BlackElo", "-"),
        "eco": clean_header("ECO", "-"),
        "opening": clean_header("Opening", "Standard Chess Game"),
        "time_control": clean_header("TimeControl", "-"),
        "termination": clean_header("Termination", "-"),
    }


def extract_moves_and_positions(game: chess.pgn.Game) -> List[Dict[str, Any]]:
    """Extract full chronological move history with board states (FENs).

    Args:
        game: Parsed chess.pgn.Game instance.

    Returns:
        List of dictionaries containing move details and resulting FEN for each ply.
        Index 0 represents the starting position (before move 1).
    """
    board = game.board()

    positions: List[Dict[str, Any]] = [
        {
            "ply": 0,
            "move_number": 0,
            "is_white": True,
            "san": "",
            "uci": "",
            "fen": board.fen(),
            "check": board.is_check(),
            "capture": False,
            "comment": "",
        }
    ]

    ply = 1
    node = game
    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        is_white = (board.turn == chess.WHITE)
        move_number = board.fullmove_number
        is_capture = board.is_capture(move)
        san = board.san(move)
        uci = move.uci()

        board.push(move)

        positions.append({
            "ply": ply,
            "move_number": move_number,
            "is_white": is_white,
            "san": san,
            "uci": uci,
            "fen": board.fen(),
            "check": board.is_check(),
            "capture": is_capture,
            "comment": next_node.comment,
        })

        ply += 1
        node = next_node

    return positions
