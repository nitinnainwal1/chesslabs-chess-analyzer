"""ChessLabs - Stockfish UCI Engine Interface.

Manages Stockfish UCI engine discovery, lifecycle, position evaluation,
score normalization, best move determination, and Principal Variation (PV) parsing.
"""

import os
import shutil
import time
from typing import Any, Dict, List, Optional
import chess
import chess.engine


def get_stockfish_path(custom_path: Optional[str] = None) -> Optional[str]:
    """Find a valid Stockfish binary path on the system.

    Search priority:
    1. Explicit custom_path argument (if provided)
    2. (a) STOCKFISH_PATH environment variable
    3. (b) System PATH via shutil.which("stockfish") or shutil.which("stockfish.exe")
    4. (c) Linux package path /usr/games/stockfish
    5. (d) Local project 'stockfish/' directory (stockfish.exe / stockfish)

    Args:
        custom_path: Optional user-specified path.

    Returns:
        Absolute path to Stockfish binary if found and executable, else None.
    """
    if custom_path is not None:
        if os.path.isfile(custom_path):
            return os.path.abspath(custom_path)
        return None

    # (a) STOCKFISH_PATH environment variable
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path and os.path.isfile(env_path):
        return os.path.abspath(env_path)

    # (b) shutil.which("stockfish")
    sys_sf = shutil.which("stockfish") or shutil.which("stockfish.exe")
    if sys_sf and os.path.isfile(sys_sf):
        return os.path.abspath(sys_sf)

    # (c) /usr/games/stockfish (standard Debian/Ubuntu package location)
    linux_pkg_path = "/usr/games/stockfish"
    if os.path.isfile(linux_pkg_path):
        return os.path.abspath(linux_pkg_path)

    # (d) Local stockfish/ directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base_dir, "stockfish", "stockfish.exe"),
        os.path.join(base_dir, "stockfish", "stockfish"),
        os.path.join(base_dir, "stockfish", "stockfish-windows-x86-64-universal.exe"),
        os.path.join(base_dir, "stockfish", "stockfish-windows-x86-64-avx2.exe"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return os.path.abspath(cand)

    return None


class StockfishEngine:
    """Manages Stockfish engine lifecycle and position evaluations."""

    def __init__(
        self,
        engine_path: Optional[str] = None,
        default_depth: int = 15,
        default_time_limit: float = 0.5,
    ) -> None:
        """Initialize engine manager.

        Args:
            engine_path: Optional explicit binary path.
            default_depth: Default search depth in plies (e.g. 15).
            default_time_limit: Default max analysis time per position in seconds.
        """
        self.engine_path = get_stockfish_path(engine_path)
        self.default_depth = default_depth
        self.default_time_limit = default_time_limit
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def is_available(self) -> bool:
        """Check if Stockfish executable is found on system."""
        return self.engine_path is not None and os.path.isfile(self.engine_path)

    def start(self) -> bool:
        """Start the UCI engine process.

        Returns:
            True if started successfully, False otherwise.
        """
        if self._engine is not None:
            return True

        if not self.is_available() or self.engine_path is None:
            return False

        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            return True
        except Exception:
            self._engine = None
            return False

    def close(self) -> None:
        """Safely terminate the engine process."""
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            finally:
                self._engine = None

    def __enter__(self) -> "StockfishEngine":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def evaluate_position(
        self,
        board: chess.Board,
        depth: Optional[int] = None,
        time_limit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Analyze a board position using Stockfish.

        Args:
            board: chess.Board position to analyze.
            depth: Optional analysis depth override.
            time_limit: Optional analysis time limit in seconds.

        Returns:
            Dictionary containing evaluation metrics:
            - score_cp: Centipawns score from White's perspective (+80 = +0.80 for White)
            - score_turn_cp: Centipawns score from active player's perspective
            - mate_score: Plies/moves to mate (None if not a forced mate)
            - score_formatted: String formatted score (e.g. "+0.82", "-1.50", "M2", "M-4")
            - best_move: chess.Move instance or None
            - best_move_san: SAN notation string (e.g. "Nf3")
            - best_move_uci: UCI notation string (e.g. "g1f3")
            - pv: List of chess.Move objects in the Principal Variation
            - pv_san: List of SAN string moves
            - pv_formatted: Human readable PV line (e.g. "Nf3 d5 c4 e6")
            - depth: Reached search depth
            - nodes: Number of nodes calculated
            - nodes_formatted: Formatted nodes string (e.g. "1.24M" or "850K")
            - time_taken: Elapsed time in seconds
        """
        if not self.start() or self._engine is None:
            return self._fallback_empty_eval(board)

        target_depth = depth or self.default_depth
        target_time = time_limit or self.default_time_limit

        # Handle terminal checkmate directly
        if board.is_checkmate():
            white_won = (board.turn == chess.BLACK)
            score_val = 10000 if white_won else -10000
            return {
                "score_cp": score_val,
                "score_turn_cp": -10000,
                "mate_score": 0,
                "score_formatted": "Mate" if white_won else "-Mate",
                "best_move": None,
                "best_move_san": "-",
                "best_move_uci": "-",
                "pv": [],
                "pv_san": [],
                "pv_formatted": "-",
                "depth": target_depth,
                "nodes": 0,
                "nodes_formatted": "0",
                "time_taken": 0.0,
            }

        # Handle stalemate / draw states directly
        if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition(3):
            return {
                "score_cp": 0,
                "score_turn_cp": 0,
                "mate_score": None,
                "score_formatted": "0.00",
                "best_move": None,
                "best_move_san": "-",
                "best_move_uci": "-",
                "pv": [],
                "pv_san": [],
                "pv_formatted": "-",
                "depth": target_depth,
                "nodes": 0,
                "nodes_formatted": "0",
                "time_taken": 0.0,
            }

        limit = chess.engine.Limit(depth=target_depth, time=target_time)

        start_t = time.time()
        try:
            info = self._engine.analyse(board, limit)
        except Exception:
            return self._fallback_empty_eval(board)

        elapsed = max(0.001, time.time() - start_t)

        # 1. Parse Score
        score_obj = info.get("score")
        score_cp = 0
        score_turn_cp = 0
        mate_score = None
        score_formatted = "0.00"

        if score_obj is not None:
            # Score from White's perspective (standard for evaluation bars & graphs)
            pov_score_white = score_obj.white()
            # Score from active player's turn perspective
            pov_score_turn = score_obj.pov(board.turn)

            if pov_score_white.is_mate():
                mate_score = pov_score_white.mate()
                if mate_score is not None:
                    score_formatted = f"M{mate_score}" if mate_score > 0 else f"M{mate_score}"
                    score_cp = 10000 if mate_score > 0 else -10000
            else:
                score_cp = pov_score_white.score(mate_score=10000) or 0
                cp_val = score_cp / 100.0
                score_formatted = f"{cp_val:+.2f}"

            if pov_score_turn.is_mate():
                turn_mate = pov_score_turn.mate()
                score_turn_cp = 10000 if (turn_mate and turn_mate > 0) else -10000
            else:
                score_turn_cp = pov_score_turn.score(mate_score=10000) or 0

        # 2. Parse Best Move and PV
        pv_moves: List[chess.Move] = info.get("pv", [])
        best_move: Optional[chess.Move] = pv_moves[0] if pv_moves else None

        best_move_san = ""
        best_move_uci = ""
        pv_san_list: List[str] = []

        if best_move is not None:
            try:
                best_move_san = board.san(best_move)
                best_move_uci = best_move.uci()
            except Exception:
                pass

        # Convert PV moves to SAN
        temp_board = board.copy()
        for m in pv_moves[:8]:  # Keep top 8 moves for PV summary
            if m in temp_board.legal_moves:
                pv_san_list.append(temp_board.san(m))
                temp_board.push(m)
            else:
                break

        pv_formatted = " ".join(pv_san_list)

        # 3. Depth & Nodes
        depth_reached = info.get("depth", target_depth)
        nodes_count = info.get("nodes", 0)

        if nodes_count >= 1_000_000:
            nodes_formatted = f"{nodes_count / 1_000_000:.2f}M"
        elif nodes_count >= 1_000:
            nodes_formatted = f"{nodes_count / 1_000:.1f}K"
        else:
            nodes_formatted = str(nodes_count)

        return {
            "score_cp": score_cp,
            "score_turn_cp": score_turn_cp,
            "mate_score": mate_score,
            "score_formatted": score_formatted,
            "best_move": best_move,
            "best_move_san": best_move_san,
            "best_move_uci": best_move_uci,
            "pv": pv_moves,
            "pv_san": pv_san_list,
            "pv_formatted": pv_formatted,
            "depth": depth_reached,
            "nodes": nodes_count,
            "nodes_formatted": nodes_formatted,
            "time_taken": round(elapsed, 3),
        }

    def _fallback_empty_eval(self, board: chess.Board) -> Dict[str, Any]:
        """Fallback evaluation dictionary if engine is unavailable."""
        return {
            "score_cp": 0,
            "score_turn_cp": 0,
            "mate_score": None,
            "score_formatted": "0.00",
            "best_move": None,
            "best_move_san": "-",
            "best_move_uci": "-",
            "pv": [],
            "pv_san": [],
            "pv_formatted": "-",
            "depth": 0,
            "nodes": 0,
            "nodes_formatted": "0",
            "time_taken": 0.0,
        }
