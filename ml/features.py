"""ChessLabs - Machine Learning Feature Extraction.

Extracts domain-engineered chess features from a board state and candidate move.
Features capture tactical complexity, material balance, king safety, center control,
and positional dynamics.
"""

from typing import Any, Dict, List
import chess

# Standard piece values for material calculations
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]
EXTENDED_CENTER = [
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
]


def calculate_material(board: chess.Board) -> Dict[str, int]:
    """Calculate White material, Black material, and material balance."""
    white_mat = 0
    black_mat = 0

    for square, piece in board.piece_map().items():
        val = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            white_mat += val
        else:
            black_mat += val

    return {
        "white_material": white_mat,
        "black_material": black_mat,
        "material_balance": white_mat - black_mat,
        "total_material": white_mat + black_mat,
    }


def calculate_center_control(board: chess.Board) -> Dict[str, int]:
    """Count attacks and occupancy in key center squares."""
    white_attacks = 0
    black_attacks = 0

    for sq in CENTER_SQUARES:
        white_attacks += len(board.attackers(chess.WHITE, sq))
        black_attacks += len(board.attackers(chess.BLACK, sq))

    return {
        "white_center_control": white_attacks,
        "black_center_control": black_attacks,
        "center_control_diff": white_attacks - black_attacks,
    }


def calculate_king_safety(board: chess.Board, color: chess.Color) -> int:
    """Evaluate king safety based on surrounding enemy attackers and friendly defenders."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0

    # Get adjacent squares around king
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)

    danger_score = 0
    opp_color = not color

    for df in [-1, 0, 1]:
        for dr in [-1, 0, 1]:
            f = king_file + df
            r = king_rank + dr
            if 0 <= f <= 7 and 0 <= r <= 7:
                target_sq = chess.square(f, r)
                # Count enemy attackers pointing at king's neighborhood
                danger_score += len(board.attackers(opp_color, target_sq))

    return danger_score


def extract_features(
    board: chess.Board,
    move: chess.Move,
    eval_before_cp: int = 0,
) -> Dict[str, Any]:
    """Extract a complete feature dictionary for a position and candidate move.

    Args:
        board: chess.Board state before the move is played.
        move: chess.Move that is played.
        eval_before_cp: Stockfish evaluation before move in centipawns (from White's POV).

    Returns:
        Dictionary of numeric and boolean features suitable for ML modeling.
    """
    is_white = (board.turn == chess.WHITE)
    mat_info = calculate_material(board)
    center_info = calculate_center_control(board)

    # Moving piece details
    piece = board.piece_at(move.from_square)
    piece_type = piece.piece_type if piece else chess.PAWN
    captured_piece = board.piece_at(move.to_square)
    captured_value = PIECE_VALUES.get(captured_piece.piece_type, 0) if captured_piece else 0

    # Board dynamics before move
    legal_moves_count = board.legal_moves.count()
    in_check_before = board.is_check()

    # Apply move hypothetically on copy to measure outcome features
    is_capture = board.is_capture(move)
    is_castling = board.is_castling(move)
    is_promotion = (move.promotion is not None)

    temp_board = board.copy()
    gives_check = False
    if move in temp_board.legal_moves:
        temp_board.push(move)
        gives_check = temp_board.is_check()

    # King safety
    white_king_danger = calculate_king_safety(board, chess.WHITE)
    black_king_danger = calculate_king_safety(board, chess.BLACK)
    user_king_danger = white_king_danger if is_white else black_king_danger
    opp_king_danger = black_king_danger if is_white else white_king_danger

    # Perspective evaluation
    eval_turn_cp = eval_before_cp if is_white else -eval_before_cp

    return {
        # Positional & Turn
        "is_white_turn": 1 if is_white else 0,
        "move_number": board.fullmove_number,
        "legal_moves_count": legal_moves_count,
        # Material
        "material_balance": mat_info["material_balance"] if is_white else -mat_info["material_balance"],
        "total_material": mat_info["total_material"],
        # Center Control
        "user_center_control": center_info["white_center_control"] if is_white else center_info["black_center_control"],
        "opp_center_control": center_info["black_center_control"] if is_white else center_info["white_center_control"],
        # King Safety
        "user_king_danger": user_king_danger,
        "opp_king_danger": opp_king_danger,
        # Move Characteristics
        "piece_type": piece_type,
        "is_capture": 1 if is_capture else 0,
        "captured_value": captured_value,
        "is_castling": 1 if is_castling else 0,
        "is_promotion": 1 if is_promotion else 0,
        "in_check_before": 1 if in_check_before else 0,
        "gives_check": 1 if gives_check else 0,
        # Engine Signal
        "eval_before_cp": eval_turn_cp,
    }


FEATURE_COLUMNS = [
    "is_white_turn",
    "move_number",
    "legal_moves_count",
    "material_balance",
    "total_material",
    "user_center_control",
    "opp_center_control",
    "user_king_danger",
    "opp_king_danger",
    "piece_type",
    "is_capture",
    "captured_value",
    "is_castling",
    "is_promotion",
    "in_check_before",
    "gives_check",
    "eval_before_cp",
]
