"""ChessLabs - Move Classification & Accuracy Engine.

Calculates Centipawn Loss (CPL) from the correct player perspective and
classifies moves into standard chess review categories:
Good Moves:
- Brilliant (!! Sacrifice leading to winning position)
- Great Move (! Standout high-impact find)
- Best (★ Engine best move or loss <= 10 cp)
- Excellent (👍 Loss 11 - 25 cp)
- Good (✓ Loss 26 - 55 cp)

Mistakes:
- Miss (✖ Missed tactical win or mate)
- Inaccuracy (?! Loss 56 - 120 cp)
- Mistake (? Loss 121 - 250 cp)
- Blunder (?? Loss > 250 cp)
"""

import math
from typing import Any, Dict, Optional
import chess


# Classification Constants
CLASS_BRILLIANT = "Brilliant"
CLASS_GREAT = "Great Move"
CLASS_BEST = "Best"
CLASS_EXCELLENT = "Excellent"
CLASS_GOOD = "Good"
CLASS_MISS = "Miss"
CLASS_INACCURACY = "Inaccuracy"
CLASS_MISTAKE = "Mistake"
CLASS_BLUNDER = "Blunder"

GOOD_MOVE_CLASSES = [CLASS_BRILLIANT, CLASS_GREAT, CLASS_BEST, CLASS_EXCELLENT, CLASS_GOOD]
MISTAKE_CLASSES = [CLASS_MISS, CLASS_INACCURACY, CLASS_MISTAKE, CLASS_BLUNDER]

# Color Codes for UI Badges & Highlights
COLOR_MAP = {
    CLASS_BRILLIANT: "#1baca6",   # Teal / Cyan
    CLASS_GREAT: "#29b6f6",       # Bright Cyan
    CLASS_BEST: "#7bb03e",        # Emerald Green
    CLASS_EXCELLENT: "#95b842",   # Lime Green
    CLASS_GOOD: "#9fa8a3",        # Grey-green
    CLASS_MISS: "#e74c3c",        # Red
    CLASS_INACCURACY: "#f5b041",  # Amber / Yellow
    CLASS_MISTAKE: "#e67e22",     # Orange
    CLASS_BLUNDER: "#c0392b",     # Crimson Red
}

# Icon Badges for UI Tables
EMOJI_MAP = {
    CLASS_BRILLIANT: "‼️",
    CLASS_GREAT: "!",
    CLASS_BEST: "⭐",
    CLASS_EXCELLENT: "👍",
    CLASS_GOOD: "✔️",
    CLASS_MISS: "❌",
    CLASS_INACCURACY: "⁉️",
    CLASS_MISTAKE: "❓",
    CLASS_BLUNDER: "⁇",
}

# Text Badges for Renderers
BADGE_TEXT_MAP = {
    CLASS_BRILLIANT: "!!",
    CLASS_GREAT: "!",
    CLASS_BEST: "★",
    CLASS_EXCELLENT: "👍",
    CLASS_GOOD: "✓",
    CLASS_MISS: "✖",
    CLASS_INACCURACY: "?!",
    CLASS_MISTAKE: "?",
    CLASS_BLUNDER: "??",
}


def is_material_sacrifice(board_before: chess.Board, move: chess.Move) -> bool:
    """Detect if a move represents a genuine tactical material sacrifice.
    
    Checks if a player intentionally gives up or risks higher-value material
    (Knight, Bishop, Rook, Queen) for positional/tactical compensation.
    """
    piece_moved = board_before.piece_at(move.from_square)
    if not piece_moved or piece_moved.piece_type in (chess.PAWN, chess.KING):
        return False

    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    moved_val = PIECE_VALUES.get(piece_moved.piece_type, 0)
    captured_piece = board_before.piece_at(move.to_square)
    captured_val = PIECE_VALUES.get(captured_piece.piece_type, 0) if captured_piece else 0

    opp_color = not piece_moved.color

    # Simulate position after the move
    board_after = board_before.copy()
    board_after.push(move)

    # 1. Higher value piece captures lower value piece while landing on attacked square
    if moved_val > captured_val:
        if board_after.is_attacked_by(opp_color, move.to_square):
            return True

    # 2. Piece moves to attacked square without equal capture
    if moved_val >= 3 and captured_val < moved_val:
        if board_after.is_attacked_by(opp_color, move.to_square):
            attackers = list(board_after.attackers(opp_color, move.to_square))
            defenders = list(board_after.attackers(piece_moved.color, move.to_square))
            if len(attackers) > len(defenders) or len(defenders) == 0:
                return True
            for att_sq in attackers:
                att_piece = board_after.piece_at(att_sq)
                if att_piece and PIECE_VALUES.get(att_piece.piece_type, 0) < moved_val:
                    return True

    return False


def calculate_centipawn_loss(
    eval_before_turn: int,
    eval_after_turn: int,
    is_mate_before: bool = False,
    is_mate_after: bool = False,
) -> int:
    """Calculate centipawn loss from the moving player's turn perspective.

    Both inputs must be from the perspective of the player whose turn it is:
    - Positive score = Advantage for moving player.
    - Loss = max(0, eval_before - eval_after).

    Args:
        eval_before_turn: Score in centipawns before move (from moving player's POV).
        eval_after_turn: Score in centipawns after move (from moving player's POV).
        is_mate_before: Whether a forced mate existed before the move.
        is_mate_after: Whether a forced mate exists after the move.

    Returns:
        Non-negative integer centipawn loss.
    """
    # If moving player had mate but lost it, that is a severe blunder
    if is_mate_before and not is_mate_after and eval_before_turn > 0:
        return 600

    # If moving player delivered mate or preserved a winning mate
    if is_mate_after and eval_after_turn >= 9000:
        return 0

    # Cap extreme centipawn swings to avoid infinity skew
    capped_before = max(-1000, min(1000, eval_before_turn))
    capped_after = max(-1000, min(1000, eval_after_turn))

    loss = capped_before - capped_after
    return max(0, loss)


def classify_move(
    is_best_move: bool,
    centipawn_loss: int,
    is_sacrifice: bool = False,
    eval_after_turn: int = 0,
    eval_before_turn: int = 0,
    is_mate_before: bool = False,
    is_mate_after: bool = False,
) -> str:
    """Classify a move into standard Chess.com game review tiers.

    Args:
        is_best_move: True if the played move matches Stockfish's top recommended move.
        centipawn_loss: Non-negative centipawn loss.
        is_sacrifice: True if the move sacrificed material.
        eval_after_turn: Centipawn eval after move from player POV.
        eval_before_turn: Centipawn eval before move from player POV.
        is_mate_before: True if position had forced mate before move.
        is_mate_after: True if position has forced mate after move.

    Returns:
        One of Brilliant, Great Move, Best, Excellent, Good, Miss, Inaccuracy, Mistake, Blunder.
    """
    # 1. Brilliant: Top engine move that sacrifices material and preserves winning advantage
    if (is_best_move or centipawn_loss <= 10) and is_sacrifice and (eval_after_turn >= 120 or is_mate_after):
        return CLASS_BRILLIANT

    # 2. Great Move: Standout finding / comeback in tough position
    if is_best_move and centipawn_loss <= 5 and eval_before_turn <= -80 and eval_after_turn >= 0:
        return CLASS_GREAT

    # 3. Best Move (top engine move or loss <= 10 cp)
    if is_best_move or centipawn_loss <= 10 or (is_mate_after and eval_after_turn >= 9000):
        return CLASS_BEST

    # 4. Excellent Move (loss 11 - 25 cp)
    if centipawn_loss <= 25:
        return CLASS_EXCELLENT

    # 5. Good Move (loss 26 - 55 cp)
    if centipawn_loss <= 55:
        return CLASS_GOOD

    # 6. Miss: Had winning advantage (>= 150 cp or mate) but threw away the tactic
    if (eval_before_turn >= 150 or is_mate_before) and eval_after_turn <= 60 and centipawn_loss >= 120:
        return CLASS_MISS

    # 7. Inaccuracy (loss 56 - 120 cp)
    if centipawn_loss <= 120:
        return CLASS_INACCURACY

    # 8. Mistake (loss 121 - 250 cp)
    if centipawn_loss <= 250:
        return CLASS_MISTAKE

    # 9. Blunder (loss > 250 cp)
    return CLASS_BLUNDER


def calculate_move_accuracy(centipawn_loss: int) -> float:
    """Calculate a single move accuracy percentage (0.0% to 100.0%).

    Uses a calibrated exponential model: Accuracy = 100 * exp(-0.0035 * CPL).

    Args:
        centipawn_loss: Centipawns lost on the move.

    Returns:
        Float accuracy percentage between 0.0 and 100.0.
    """
    loss = max(0, centipawn_loss)
    acc = 100.0 * math.exp(-0.0035 * loss)
    return round(max(0.0, min(100.0, acc)), 1)
