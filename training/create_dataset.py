"""ChessLabs - ML Dataset Creation Pipeline.

Processes PGN game files, runs Stockfish evaluations on each ply to determine
centipawn losses and blunder labels, extracts engineered chess features,
and outputs a clean CSV dataset for ML training with game_id to prevent data leakage.
"""

import os
import sys

# Ensure workspace root is in path before importing internal packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
from typing import Any, Dict, List
import pandas as pd
import chess
import chess.pgn

from chess.stockfish_engine import StockfishEngine
from chess.move_analyzer import calculate_centipawn_loss
from ml.features import extract_features, FEATURE_COLUMNS


def build_dataset_from_pgn_files(
    pgn_files: List[str],
    output_csv_path: str,
    depth: int = 10,
    time_limit: float = 0.1,
) -> pd.DataFrame:
    """Process multiple PGN files and build a machine learning dataset.

    Args:
        pgn_files: List of file paths to .pgn files.
        output_csv_path: Destination path for saving the CSV dataset.
        depth: Stockfish analysis depth.
        time_limit: Stockfish time limit per position.

    Returns:
        pandas.DataFrame containing all extracted positions and labels.
    """
    rows: List[Dict[str, Any]] = []
    game_counter = 1

    print(f"Initializing Stockfish for dataset generation (depth={depth})...")
    with StockfishEngine(default_depth=depth, default_time_limit=time_limit) as engine:
        for pgn_file in pgn_files:
            if not os.path.isfile(pgn_file):
                print(f"Skipping missing file: {pgn_file}")
                continue

            print(f"Processing PGN file: {os.path.basename(pgn_file)}...")
            with open(pgn_file, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break

                    board = game.board()
                    moves = list(game.mainline_moves())
                    if not moves:
                        continue

                    # Starting position eval
                    prev_eval = engine.evaluate_position(board, depth=depth, time_limit=time_limit)

                    for ply_idx, move in enumerate(moves, start=1):
                        is_white = (board.turn == chess.WHITE)
                        eval_before_cp = prev_eval["score_cp"]
                        mate_before = prev_eval["mate_score"] is not None

                        # Extract engineered features before move is pushed
                        feature_dict = extract_features(board, move, eval_before_cp=eval_before_cp)

                        # Push move and evaluate resulting position
                        board.push(move)
                        curr_eval = engine.evaluate_position(board, depth=depth, time_limit=time_limit)

                        eval_before_turn = eval_before_cp if is_white else -eval_before_cp
                        eval_after_turn = curr_eval["score_cp"] if is_white else -curr_eval["score_cp"]
                        mate_after = curr_eval["mate_score"] is not None

                        cpl = calculate_centipawn_loss(
                            eval_before_turn=eval_before_turn,
                            eval_after_turn=eval_after_turn,
                            is_mate_before=mate_before,
                            is_mate_after=mate_after,
                        )

                        # Blunder label: 1 if CPL >= 200 cp, else 0
                        is_blunder = 1 if cpl >= 200 else 0

                        # Multi-class label: 0 (Best/Good), 1 (Inaccuracy), 2 (Mistake), 3 (Blunder)
                        if cpl <= 45:
                            quality_class = 0
                        elif cpl <= 110:
                            quality_class = 1
                        elif cpl <= 220:
                            quality_class = 2
                        else:
                            quality_class = 3

                        row = {
                            "game_id": game_counter,
                            "ply": ply_idx,
                            "fen": board.fen(),
                            "eval_after": curr_eval["score_cp"],
                            "eval_before": eval_before_cp,
                            **feature_dict,
                            "centipawn_loss": cpl,
                            "is_blunder": is_blunder,
                            "quality_class": quality_class,
                        }
                        rows.append(row)
                        prev_eval = curr_eval

                    game_counter += 1

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"[SUCCESS] Generated dataset with {len(df)} positions across {game_counter-1} games. Saved to {output_csv_path}")
    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_files = [
        os.path.join(base_dir, "data", "sample", "opera_game.pgn"),
        os.path.join(base_dir, "data", "sample", "immortal_game.pgn"),
        os.path.join(base_dir, "data", "sample", "scholars_mate.pgn"),
        os.path.join(base_dir, "data", "sample", "master_games.pgn"),
    ]
    out_csv = os.path.join(base_dir, "data", "processed", "chess_ml_dataset.csv")
    build_dataset_from_pgn_files(sample_files, out_csv, depth=8, time_limit=0.08)
