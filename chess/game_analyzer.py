"""ChessLabs - Full Game Analyzer.

Evaluates an entire chess game move-by-move with Stockfish,
computes centipawn losses, move classifications, White/Black accuracies,
and identifies key turning points and critical moments.
"""

from typing import Any, Callable, Dict, List, Optional
import chess
import chess.pgn

from chess.stockfish_engine import StockfishEngine
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
    GOOD_MOVE_CLASSES,
    MISTAKE_CLASSES,
    COLOR_MAP,
    EMOJI_MAP,
    BADGE_TEXT_MAP,
)
from chess.pgn_parser import extract_game_metadata


def analyze_full_game(
    game: chess.pgn.Game,
    engine: StockfishEngine,
    depth: int = 12,
    time_limit: float = 0.2,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Run full move-by-move Stockfish analysis across a parsed chess game.

    Args:
        game: Parsed chess.pgn.Game.
        engine: Active StockfishEngine instance.
        depth: Analysis depth per position.
        time_limit: Max time allocated per position (seconds).
        progress_callback: Optional callback receiving (current_ply, total_plies).

    Returns:
        Analysis summary dictionary containing:
        - metadata: Game header metadata
        - positions: List of analyzed position dictionaries with evaluations
        - white_accuracy: Float (0-100%)
        - black_accuracy: Float (0-100%)
        - white_counts: Dict of move classifications for White
        - black_counts: Dict of move classifications for Black
        - critical_moments: List of plies with high evaluation swings (>150 cp)
    """
    metadata = extract_game_metadata(game)
    board = game.board()
    moves = list(game.mainline_moves())
    total_plies = len(moves)

    # 1. Analyze starting position (Ply 0)
    initial_eval = engine.evaluate_position(board, depth=depth, time_limit=time_limit)

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
            "eval_white": initial_eval["score_cp"],
            "eval_formatted": initial_eval["score_formatted"],
            "mate_score": initial_eval["mate_score"],
            "best_move": initial_eval["best_move"],
            "best_move_san": initial_eval["best_move_san"],
            "pv_formatted": initial_eval["pv_formatted"],
            "depth": initial_eval["depth"],
            "cpl": 0,
            "classification": CLASS_BEST,
            "color_code": COLOR_MAP[CLASS_BEST],
            "emoji": EMOJI_MAP[CLASS_BEST],
            "accuracy": 100.0,
        }
    ]

    white_losses: List[int] = []
    black_losses: List[int] = []

    white_counts = {
        CLASS_BRILLIANT: 0,
        CLASS_GREAT: 0,
        CLASS_BEST: 0,
        CLASS_EXCELLENT: 0,
        CLASS_GOOD: 0,
        CLASS_MISS: 0,
        CLASS_INACCURACY: 0,
        CLASS_MISTAKE: 0,
        CLASS_BLUNDER: 0,
    }
    black_counts = {
        CLASS_BRILLIANT: 0,
        CLASS_GREAT: 0,
        CLASS_BEST: 0,
        CLASS_EXCELLENT: 0,
        CLASS_GOOD: 0,
        CLASS_MISS: 0,
        CLASS_INACCURACY: 0,
        CLASS_MISTAKE: 0,
        CLASS_BLUNDER: 0,
    }

    critical_moments: List[Dict[str, Any]] = []

    # 2. Iterate through each ply
    prev_eval = initial_eval
    for ply_idx, move in enumerate(moves, start=1):
        is_white = (board.turn == chess.WHITE)
        move_number = board.fullmove_number
        is_capture = board.is_capture(move)
        san = board.san(move)
        uci = move.uci()

        # Check if piece was sacrificed
        is_sacrifice = is_material_sacrifice(board, move)

        # Was played move matching engine's best move before push?
        is_best = (prev_eval["best_move"] is not None and move == prev_eval["best_move"])
        best_san_before = prev_eval["best_move_san"]

        # Push move to reach new position
        board.push(move)

        # Evaluate resulting position
        curr_eval = engine.evaluate_position(board, depth=depth, time_limit=time_limit)

        # Calculate centipawn loss from moving player's perspective
        # If White moved: score from White's POV before vs after
        # If Black moved: score from Black's POV before vs after (-score_white)
        eval_before_turn = prev_eval["score_cp"] if is_white else -prev_eval["score_cp"]
        eval_after_turn = curr_eval["score_cp"] if is_white else -curr_eval["score_cp"]

        mate_before = prev_eval["mate_score"] is not None
        mate_after = curr_eval["mate_score"] is not None

        cpl = calculate_centipawn_loss(
            eval_before_turn=eval_before_turn,
            eval_after_turn=eval_after_turn,
            is_mate_before=mate_before,
            is_mate_after=mate_after,
        )

        classification = classify_move(
            is_best_move=is_best,
            centipawn_loss=cpl,
            is_sacrifice=is_sacrifice,
            eval_after_turn=eval_after_turn,
            eval_before_turn=eval_before_turn,
            is_mate_before=mate_before,
            is_mate_after=mate_after,
        )
        ply_accuracy = calculate_move_accuracy(cpl)

        # Track stats
        if is_white:
            white_losses.append(cpl)
            white_counts[classification] += 1
        else:
            black_losses.append(cpl)
            black_counts[classification] += 1

        # Check for critical turning moment (swing >= 160 cp or blunder)
        if cpl >= 160 or classification == CLASS_BLUNDER:
            critical_moments.append({
                "ply": ply_idx,
                "move_number": move_number,
                "player": "White" if is_white else "Black",
                "move": san,
                "best_move": best_san_before,
                "cpl": cpl,
                "classification": classification,
                "eval_after": curr_eval["score_formatted"],
            })

        positions.append({
            "ply": ply_idx,
            "move_number": move_number,
            "is_white": is_white,
            "san": san,
            "uci": uci,
            "fen": board.fen(),
            "check": board.is_check(),
            "capture": is_capture,
            "eval_white": curr_eval["score_cp"],
            "eval_formatted": curr_eval["score_formatted"],
            "mate_score": curr_eval["mate_score"],
            "best_move": curr_eval["best_move"],
            "best_move_san": curr_eval["best_move_san"],
            "best_move_before": best_san_before,
            "pv_formatted": curr_eval["pv_formatted"],
            "depth": curr_eval["depth"],
            "cpl": cpl,
            "classification": classification,
            "color_code": COLOR_MAP[classification],
            "emoji": EMOJI_MAP[classification],
            "accuracy": ply_accuracy,
        })

        prev_eval = curr_eval

        if progress_callback:
            progress_callback(ply_idx, total_plies)

    # 3. Overall Accuracies
    def avg_accuracy(losses: List[int]) -> float:
        if not losses:
            return 100.0
        accs = [calculate_move_accuracy(l) for l in losses]
        return round(sum(accs) / len(accs), 1)

    white_acc = avg_accuracy(white_losses)
    black_acc = avg_accuracy(black_losses)
    white_cpl_avg = round(sum(white_losses) / len(white_losses), 1) if white_losses else 0.0
    black_cpl_avg = round(sum(black_losses) / len(black_losses), 1) if black_losses else 0.0

    return {
        "metadata": metadata,
        "positions": positions,
        "total_plies": total_plies,
        "white_accuracy": white_acc,
        "black_accuracy": black_acc,
        "white_cpl_avg": white_cpl_avg,
        "black_cpl_avg": black_cpl_avg,
        "white_counts": white_counts,
        "black_counts": black_counts,
        "critical_moments": critical_moments,
    }


def create_evaluation_chart(positions: List[Dict[str, Any]], current_ply: int = 0) -> Any:
    """Generate an interactive Plotly evaluation curve with blunder markers and active ply cursor.

    Args:
        positions: Chronological list of position evaluation dictionaries.
        current_ply: Currently active ply to highlight with a marker/line.

    Returns:
        plotly.graph_objects.Figure object.
    """
    import plotly.graph_objects as go

    x_indices = []
    x_labels = []
    y_evals = []
    hover_texts = []

    blunder_x = []
    blunder_y = []
    blunder_hover = []

    mistake_x = []
    mistake_y = []
    mistake_hover = []

    for pos in positions:
        ply = pos["ply"]
        x_indices.append(ply)

        if ply == 0:
            label = "Start"
        else:
            turn_dot = "." if pos["is_white"] else "..."
            label = f"{pos['move_number']}{turn_dot} {pos['san']}"
        x_labels.append(label)

        # Convert centipawns to clamped pawns (-8.0 to +8.0)
        eval_cp = pos.get("eval_white", 0)
        mate = pos.get("mate_score")
        if mate is not None:
            pawn_val = 8.0 if mate > 0 else -8.0
        else:
            pawn_val = max(-8.0, min(8.0, eval_cp / 100.0))
        y_evals.append(pawn_val)

        # Hover text details
        classif = pos.get("classification", "Best")
        cpl = pos.get("cpl", 0)
        best_rec = pos.get("best_move_san", "-")
        eval_str = pos.get("eval_formatted", "0.00")
        player_name = "White" if pos["is_white"] else "Black"

        hover = (
            f"<b>{label}</b> ({player_name})<br>"
            f"Evaluation: <b>{eval_str}</b><br>"
            f"Move Quality: <b>{classif}</b> (-{cpl} cp)<br>"
            f"Best Move: {best_rec}"
        )
        hover_texts.append(hover)

        if classif == CLASS_BLUNDER and ply > 0:
            blunder_x.append(ply)
            blunder_y.append(pawn_val)
            blunder_hover.append(f"💥 <b>BLUNDER</b> on {label}<br>Loss: -{cpl} cp<br>Eval: {eval_str}")

        if classif == CLASS_MISTAKE and ply > 0:
            mistake_x.append(ply)
            mistake_y.append(pawn_val)
            mistake_hover.append(f"❓ <b>MISTAKE</b> on {label}<br>Loss: -{cpl} cp<br>Eval: {eval_str}")

    fig = go.Figure()

    # 1. Main Smooth Evaluation Curve
    fig.add_trace(
        go.Scatter(
            x=x_indices,
            y=y_evals,
            mode="lines+markers",
            line=dict(color="#d4af37", width=3, shape="spline", smoothing=0.7),
            marker=dict(size=4, color="#d4af37"),
            hovertext=hover_texts,
            hoverinfo="text",
            name="Evaluation",
            fill="tozeroy",
            fillcolor="rgba(212, 175, 55, 0.12)",
        )
    )

    # 2. Mistake Markers
    if mistake_x:
        fig.add_trace(
            go.Scatter(
                x=mistake_x,
                y=mistake_y,
                mode="markers",
                marker=dict(symbol="diamond", size=10, color="#e67e22", line=dict(width=1, color="#fff")),
                hovertext=mistake_hover,
                hoverinfo="text",
                name="Mistake",
            )
        )

    # 3. Blunder Markers
    if blunder_x:
        fig.add_trace(
            go.Scatter(
                x=blunder_x,
                y=blunder_y,
                mode="markers",
                marker=dict(symbol="x", size=12, color="#e74c3c", line=dict(width=2, color="#e74c3c")),
                hovertext=blunder_hover,
                hoverinfo="text",
                name="Blunder",
            )
        )

    # 4. Highlight Active Current Move
    if 0 <= current_ply < len(y_evals):
        curr_val = y_evals[current_ply]
        fig.add_trace(
            go.Scatter(
                x=[current_ply],
                y=[curr_val],
                mode="markers",
                marker=dict(symbol="circle", size=14, color="#3498db", line=dict(width=2, color="#ffffff")),
                hoverinfo="none",
                showlegend=False,
                name="Current Move",
            )
        )

    # Layout styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#181818",
        plot_bgcolor="#181818",
        margin=dict(l=30, r=20, t=30, b=30),
        height=280,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            title="Ply Number",
            gridcolor="#282828",
            zeroline=False,
            showticklabels=True,
        ),
        yaxis=dict(
            title="Advantage (Pawns)",
            range=[-8.5, 8.5],
            gridcolor="#282828",
            zeroline=True,
            zerolinecolor="#555555",
            zerolinewidth=1.5,
            tickvals=[-8, -4, 0, 4, 8],
            ticktext=["-8 (Black)", "-4", "0.0", "+4", "+8 (White)"],
        ),
        hovermode="closest",
    )

    return fig


def create_move_breakdown_chart(white_counts: Dict[str, int], black_counts: Dict[str, int]) -> Any:
    """Generate a grouped bar chart comparing move classifications."""
    import plotly.graph_objects as go

    all_categories = [
        CLASS_BRILLIANT,
        CLASS_GREAT,
        CLASS_BEST,
        CLASS_EXCELLENT,
        CLASS_GOOD,
        CLASS_MISS,
        CLASS_INACCURACY,
        CLASS_MISTAKE,
        CLASS_BLUNDER,
    ]
    # Filter to categories present or standard ones
    categories = [
        c for c in all_categories
        if (white_counts.get(c, 0) > 0 or black_counts.get(c, 0) > 0 or c in (CLASS_BEST, CLASS_GOOD, CLASS_INACCURACY, CLASS_MISTAKE, CLASS_BLUNDER))
    ]

    w_vals = [white_counts.get(c, 0) for c in categories]
    b_vals = [black_counts.get(c, 0) for c in categories]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="White", x=categories, y=w_vals, marker_color="#f0d9b5"))
    fig.add_trace(go.Bar(name="Black", x=categories, y=b_vals, marker_color="#769656"))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
        margin=dict(l=20, r=20, t=30, b=20),
        height=280,
        barmode="group",
        xaxis=dict(gridcolor="#2d2d2d"),
        yaxis=dict(title="Move Count", gridcolor="#2d2d2d"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

