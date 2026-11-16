"""ChessLabs - Chessboard Renderer.

Generates custom-styled SVG chessboard representations using python-chess svg rendering.
Follows the ChessLabs UI theme: Warm Ivory (#f0d9b5) & Dark Brown (#b58863),
Subtle Gold (#d4af37) move highlights, Soft Coral check indicators,
and Chess.com style Move Classification Badges directly on destination piece squares!
"""

from typing import Dict, List, Optional, Tuple, Union
import chess
import chess.svg


# ChessLabs Board Color Palette
LIGHT_SQUARE_COLOR = "#f0d9b5"  # Warm Ivory
DARK_SQUARE_COLOR = "#b58863"   # Classic Wood Dark Brown
LAST_MOVE_COLOR = "rgba(212, 175, 55, 0.55)"  # Subtle Gold highlight
CHECK_COLOR = "rgba(220, 53, 69, 0.70)"       # Crimson Coral check highlight

# Move classification circular badge configs: (fill_color, symbol, font_size, y_offset)
BADGE_CONFIG: Dict[str, Tuple[str, str, float, float]] = {
    "Brilliant": ("#1baca6", "!!", 9.5, 3.2),
    "Great Move": ("#29b6f6", "!", 10.5, 3.2),
    "Best": ("#7bb03e", "★", 9.0, 3.0),
    "Excellent": ("#95b842", "👍", 8.5, 3.0),
    "Good": ("#9fa8a3", "✓", 9.5, 3.2),
    "Miss": ("#e74c3c", "✖", 9.0, 3.0),
    "Inaccuracy": ("#f5b041", "?!", 8.8, 3.2),
    "Mistake": ("#e67e22", "?", 10.5, 3.2),
    "Blunder": ("#c0392b", "??", 8.8, 3.2),
}


def create_badge_svg_overlay(
    square: chess.Square,
    classification: str,
    flipped: bool = False,
) -> str:
    """Generate SVG markup for a circular classification badge on the target piece square.

    Args:
        square: Target chess.Square (destination of the played move).
        classification: Move classification category name.
        flipped: Whether the board is rendered from Black's perspective.

    Returns:
        SVG string with circle and text badge elements.
    """
    if classification not in BADGE_CONFIG:
        return ""

    file = chess.square_file(square)
    rank = chess.square_rank(square)
    file_idx = (7 - file) if flipped else file
    rank_idx = rank if flipped else (7 - rank)

    # Standard python-chess SVG board has margin=15 and square_size=45 in 390x390 viewBox
    # Target top-right corner of destination square
    cx = 15 + file_idx * 45 + 36.5
    cy = 15 + rank_idx * 45 + 8.5

    color, symbol, font_size, y_off = BADGE_CONFIG[classification]

    return (
        f'<g id="badge-{square}" class="move-classification-badge">'
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9.5" fill="{color}" stroke="#ffffff" stroke-width="1.6" />'
        f'<text x="{cx:.1f}" y="{(cy + y_off):.1f}" font-size="{font_size}" font-family="Segoe UI, -apple-system, Arial, sans-serif" '
        f'font-weight="900" fill="#ffffff" text-anchor="middle">{symbol}</text>'
        f'</g>'
    )


def generate_board_svg(
    board: chess.Board,
    last_move: Optional[chess.Move] = None,
    flipped: bool = False,
    arrows: Optional[List[chess.svg.Arrow]] = None,
    size: int = 420,
    classification: Optional[str] = None,
) -> str:
    """Generate a high-quality styled SVG string of the chessboard.

    Args:
        board: chess.Board instance representing the current position.
        last_move: chess.Move instance to highlight (from-square and to-square).
        flipped: If True, renders from Black's perspective; otherwise White's.
        arrows: Optional list of chess.svg.Arrow objects (e.g. for engine recommendations).
        size: Width and height of the SVG board in pixels.
        classification: Optional move classification grade for the last played move.

    Returns:
        String containing clean SVG markup.
    """
    check_square = None
    if board.is_check():
        check_square = board.king(board.turn)

    # Configure custom colors
    colors = {
        "square light": LIGHT_SQUARE_COLOR,
        "square dark": DARK_SQUARE_COLOR,
        "square light lastmove": LAST_MOVE_COLOR,
        "square dark lastmove": LAST_MOVE_COLOR,
        "margin": "#1a1a1a",
        "coord": "#a0a0a0",
    }

    svg_content = chess.svg.board(
        board=board,
        orientation=chess.BLACK if flipped else chess.WHITE,
        lastmove=last_move,
        check=check_square,
        arrows=arrows or [],
        size=size,
        colors=colors,
    )

    # Inject move classification badge overlay on the destination piece square
    if last_move and classification and classification in BADGE_CONFIG:
        badge_markup = create_badge_svg_overlay(
            square=last_move.to_square,
            classification=classification,
            flipped=flipped,
        )
        if badge_markup and "</svg>" in svg_content:
            svg_content = svg_content.replace("</svg>", f"{badge_markup}</svg>")

    return svg_content


def get_board_from_fen(fen: str) -> chess.Board:
    """Safely create a chess.Board from a FEN string with fallback.

    Args:
        fen: FEN string.

    Returns:
        chess.Board object.
    """
    try:
        return chess.Board(fen)
    except ValueError:
        return chess.Board()
