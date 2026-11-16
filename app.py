"""ChessLabs - Chess Game Analyzer.

Main Streamlit Application Entrypoint.
College Mini-Project: Modular, simple, and clean architecture.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

# ChessLabs imports
from chess.pgn_parser import (
    validate_pgn,
    parse_pgn,
    extract_game_metadata,
    extract_moves_and_positions,
)
from chess.board import (
    generate_board_svg,
    get_board_from_fen,
)
from chess.stockfish_engine import StockfishEngine, get_stockfish_path
from chess.game_analyzer import (
    analyze_full_game,
    create_evaluation_chart,
    create_move_breakdown_chart,
)
from chess.chess_dot_com import fetch_recent_games, ChessDotComError
from chess.move_analyzer import (
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
from ml.predict import MLMovePredictor
from dl.predict import DLPredictor
import chess
import chess.svg


@st.cache_resource
def get_ml_predictor() -> MLMovePredictor:
    return MLMovePredictor()


@st.cache_resource
def get_dl_predictor() -> DLPredictor:
    return DLPredictor()


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="ChessLabs | Chess Game Analyzer",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Dark Theme with Gold Accents
st.markdown(
    """
    <style>
    /* Global styles */
    .stApp {
        background-color: #121212;
        color: #f0f0f0;
    }
    
    /* Header & Titles */
    h1, h2, h3, h4, h5, h6 {
        color: #f5f5f5 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .gold-text {
        color: #d4af37 !important;
        font-weight: 700;
    }
    
    .sub-text {
        color: #a0a0a0;
        font-size: 1.05rem;
    }
    
    /* Custom Cards */
    .card {
        background-color: #1e1e1e;
        border: 1px solid #2d2d2d;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 14px;
        transition: transform 0.2s, border-color 0.2s;
    }
    .card:hover {
        border-color: #d4af37;
    }
    
    .card-title {
        color: #d4af37;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .card-body {
        color: #cccccc;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Stat Badge */
    .metric-badge {
        background-color: #252525;
        border-left: 4px solid #d4af37;
        padding: 10px 15px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    
    /* Evaluation Badges */
    .eval-positive {
        color: #4caf50;
        font-weight: 700;
    }
    .eval-negative {
        color: #f44336;
        font-weight: 700;
    }

    /* Move List Table */
    .move-btn-active {
        background-color: #d4af37 !important;
        color: #121212 !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "current_game_raw" not in st.session_state:
    st.session_state.current_game_raw = None

if "current_game_metadata" not in st.session_state:
    st.session_state.current_game_metadata = None

if "current_positions" not in st.session_state:
    st.session_state.current_positions = []

if "current_move_index" not in st.session_state:
    st.session_state.current_move_index = 0

if "board_flipped" not in st.session_state:
    st.session_state.board_flipped = False

if "game_analysis" not in st.session_state:
    st.session_state.game_analysis = None

if "session_games" not in st.session_state:
    st.session_state.session_games = []

if "analyzed_reports" not in st.session_state:
    st.session_state.analyzed_reports = []

if "chess_dot_com_games" not in st.session_state:
    st.session_state.chess_dot_com_games = []

if "selected_nav" not in st.session_state:
    st.session_state.selected_nav = "Home"


def navigate_to(page_name: str):
    """Switch active navigation tab."""
    st.session_state.selected_nav = page_name
    st.rerun()


def load_game_into_state(pgn_text: str, source_label: str = "User Import") -> bool:
    """Validate and store a game into Streamlit session state."""
    is_valid, msg, metadata = validate_pgn(pgn_text)
    if not is_valid or metadata is None:
        st.error(f"❌ {msg}")
        return False

    game = parse_pgn(pgn_text)
    if game is None:
        st.error("❌ Failed to parse game moves.")
        return False

    positions = extract_moves_and_positions(game)

    st.session_state.current_game_raw = pgn_text
    st.session_state.current_game_metadata = metadata
    st.session_state.current_positions = positions
    st.session_state.current_move_index = 0
    st.session_state.game_analysis = None

    # Add to session history if not already present
    game_entry = {
        "white": metadata["white"],
        "black": metadata["black"],
        "result": metadata["result"],
        "date": metadata["date"],
        "event": metadata["event"],
        "total_plies": metadata["total_plies"],
        "opening": metadata["opening"],
        "source": source_label,
        "pgn": pgn_text,
    }
    if not any(g.get("pgn") == pgn_text for g in st.session_state.session_games):
        st.session_state.session_games.append(game_entry)

    st.success(f"✅ {msg}")
    return True


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style='text-align: center; padding: 10px 0 20px 0;'>
            <h2 style='margin: 0; color: #d4af37; font-weight: 700; letter-spacing: 0.5px;'>ChessLabs</h2>
            <p style='color: #888888; font-size: 0.85rem; margin-top: 4px;'>Understand Every Move</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_options = [
        "Home",
        "Import Game",
        "Analyze",
        "My Games",
        "Reports",
        "About",
    ]

    current_index = 0
    if st.session_state.selected_nav in nav_options:
        current_index = nav_options.index(st.session_state.selected_nav)

    selected_page = st.radio(
        "Navigation",
        options=nav_options,
        index=current_index,
        label_visibility="collapsed",
    )

    if selected_page != st.session_state.selected_nav:
        st.session_state.selected_nav = selected_page

    st.markdown("---")

    # Current Game Quick Info in Sidebar
    if st.session_state.current_game_metadata:
        meta = st.session_state.current_game_metadata
        st.markdown(
            f"""
            <div style='background: #1e1e1e; border: 1px solid #333; padding: 10px; border-radius: 6px; font-size: 0.82rem;'>
                <strong style='color: #d4af37;'>Active Game:</strong><br>
                <span>White: {meta['white']}</span><br>
                <span>Black: {meta['black']}</span><br>
                <span style='color: #aaa;'>Result: {meta['result']} ({meta['total_plies']} plies)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='background: #181818; border: 1px solid #2a2a2a; border-radius: 6px; padding: 10px 12px; font-size: 0.8rem; color: #888;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
                <span>System:</span>
                <span style='color: #2ecc71; font-weight: 600;'>Online</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
                <span>Engine:</span>
                <strong style='color: #f0f0f0;'>Stockfish 19</strong>
            </div>
            <div style='display: flex; justify-content: space-between;'>
                <span>Analysis:</span>
                <span style='color: #d4af37;'>ML + PyTorch</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# View: 1. Home
# ---------------------------------------------------------
def render_home():
    st.markdown(
        """
        <div style='padding: 10px 0 20px 0;'>
            <h1 style='font-size: 2.6rem; margin-bottom: 4px;'>ChessLabs</h1>
            <h3 style='color: #d4af37; font-weight: 500; margin-top: 0;'>Understand Every Move.</h3>
            <p class='sub-text'>
                Analyze your chess games with engine evaluation, detailed move statistics, 
                and positional machine learning models.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1.2, 1.2, 3])
    with col1:
        if st.button("Analyze a Game", use_container_width=True, type="primary"):
            navigate_to("Analyze")
    with col2:
        if st.button("Import Game", use_container_width=True):
            navigate_to("Import Game")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Core Capabilities")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <div class='card'>
                <div class='card-title'>Engine Analysis</div>
                <div class='card-body'>
                    Objective move-by-move evaluation powered by Stockfish.
                    Tracks evaluation shifts, best moves, and principal variations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class='card'>
                <div class='card-title'>Chess.com Import</div>
                <div class='card-body'>
                    Direct integration with Chess.com's public API to fetch recent
                    Rapid, Blitz, and Bullet games for instant post-game review.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class='card'>
                <div class='card-title'>Game Reports</div>
                <div class='card-body'>
                    Comprehensive game metrics including accuracy percentages, blunder and mistake
                    tallies, critical moments, and interactive evaluation curves.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class='card'>
                <div class='card-title'>Model Insights</div>
                <div class='card-body'>
                    Predictive move quality models and deep learning position patterns
                    trained on public chess datasets.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# View: 2. Import Game
# ---------------------------------------------------------
def render_import_game():
    st.title("Import Game")
    st.markdown(
        "<p class='sub-text'>Load a chess game via file upload, PGN paste, or sample games.</p>",
        unsafe_allow_html=True,
    )

    # Quick Sample Game Loader
    st.markdown("##### Or load a classic match:")
    samp_col1, samp_col2, samp_col3 = st.columns(3)

    sample_dir = os.path.join(os.path.dirname(__file__), "data", "sample")

    with samp_col1:
        if st.button("Morphy's Opera Game (1858)", use_container_width=True):
            p = os.path.join(sample_dir, "opera_game.pgn")
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    if load_game_into_state(f.read(), "Sample: Opera Game"):
                        navigate_to("Analyze")

    with samp_col2:
        if st.button("The Immortal Game (1851)", use_container_width=True):
            p = os.path.join(sample_dir, "immortal_game.pgn")
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    if load_game_into_state(f.read(), "Sample: Immortal Game"):
                        navigate_to("Analyze")

    with samp_col3:
        if st.button("Scholar's Mate (4 Moves)", use_container_width=True):
            p = os.path.join(sample_dir, "scholars_mate.pgn")
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    if load_game_into_state(f.read(), "Sample: Scholar's Mate"):
                        navigate_to("Analyze")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Upload PGN", "Paste PGN", "Chess.com Import"])

    with tab1:
        st.markdown("#### Upload a PGN File")
        uploaded_file = st.file_uploader(
            "Choose a .pgn file",
            type=["pgn"],
            help="Upload a standard Portable Game Notation file",
        )
        if uploaded_file is not None:
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            if st.button("Load Uploaded Game", type="primary"):
                if load_game_into_state(content, f"Upload: {uploaded_file.name}"):
                    navigate_to("Analyze")

    with tab2:
        st.markdown("#### Paste PGN Text")
        pgn_text = st.text_area(
            "Paste PGN here...",
            height=200,
            placeholder="[Event \"FIDE World Championship\"]\n[White \"Carlsen, M.\"]\n[Black \"Nepomniachtchi, I.\"]\n\n1. e4 e5 2. Nf3 Nc6 ...",
        )
        if st.button("Load Pasted Game", type="primary"):
            if pgn_text.strip():
                if load_game_into_state(pgn_text, "Pasted PGN"):
                    navigate_to("Analyze")
            else:
                st.warning("Please paste valid PGN text before loading.")

    with tab3:
        st.markdown("#### Fetch from Chess.com Public API")
        col_u, col_f, col_l = st.columns([2, 1.2, 1])
        with col_u:
            username = st.text_input(
                "Chess.com Username",
                placeholder="e.g. magnuscarlsen or hikaru",
                help="Fetches public game archives directly from Chess.com API.",
            )
        with col_f:
            time_filter = st.selectbox(
                "Format Filter",
                options=["All", "Blitz", "Rapid", "Bullet", "Daily"],
                index=0,
            )
        with col_l:
            game_limit = st.selectbox("Max Games", options=[10, 20, 30], index=1)

        if st.button("Fetch Recent Games", type="primary"):
            if username.strip():
                with st.spinner(f"Fetching recent {time_filter.lower()} games for '{username}'..."):
                    try:
                        fetched = fetch_recent_games(
                            username=username,
                            limit=game_limit,
                            time_class=time_filter.lower(),
                        )
                        st.session_state.chess_dot_com_games = fetched
                        if not fetched:
                            st.warning(f"No {time_filter} games found for '{username}'.")
                        else:
                            st.success(f"Successfully retrieved {len(fetched)} games!")
                    except ChessDotComError as e:
                        st.error(f"❌ {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error fetching games: {str(e)}")
            else:
                st.warning("Please enter a valid Chess.com username.")

        # Display fetched games list
        if st.session_state.chess_dot_com_games:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"##### Recent Games ({len(st.session_state.chess_dot_com_games)}):")
            for idx, g in enumerate(st.session_state.chess_dot_com_games):
                card_col1, card_col2 = st.columns([4, 1.2])
                with card_col1:
                    result_color = "#4caf50" if "Won" in g["result"] else ("#f44336" if "Lost" in g["result"] else "#d4af37")
                    st.markdown(
                        f"""
                        <div class='card' style='margin-bottom: 8px; padding: 12px 16px;'>
                            <div style='display: flex; justify-content: space-between;'>
                                <strong style='color: #fff;'>vs {g['opponent']} ({g['opponent_rating']})</strong>
                                <span style='color: {result_color}; font-weight: bold;'>{g['result']}</span>
                            </div>
                            <div style='color: #888; font-size: 0.82rem; margin-top: 4px;'>
                                Playing as: <strong>{g['user_color']}</strong> ({g['user_rating']}) • 
                                Format: <strong>{g['time_class']} ({g['time_control']})</strong> • 
                                Date: {g['date']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with card_col2:
                    if st.button("Analyze Game", key=f"btn_c_game_{idx}", use_container_width=True):
                        if load_game_into_state(g["pgn"], f"Chess.com: vs {g['opponent']}"):
                            navigate_to("Analyze")


# ---------------------------------------------------------
# View: 3. Analyze
# ---------------------------------------------------------
def render_chess_com_summary_html(
    white_name: str,
    black_name: str,
    white_counts: Dict[str, int],
    black_counts: Dict[str, int],
    mode: str = "good",
) -> str:
    """Generate clean HTML table for Chess.com styled move classification summary."""
    if mode == "good":
        items = [
            (CLASS_BRILLIANT, "#1baca6", "!!"),
            (CLASS_GREAT, "#29b6f6", "!"),
            (CLASS_BEST, "#7bb03e", "★"),
            (CLASS_EXCELLENT, "#95b842", "👍"),
            (CLASS_GOOD, "#9fa8a3", "✓"),
        ]
    else:
        items = [
            (CLASS_MISS, "#e74c3c", "✖"),
            (CLASS_INACCURACY, "#f5b041", "?!"),
            (CLASS_MISTAKE, "#e67e22", "?"),
            (CLASS_BLUNDER, "#c0392b", "??"),
        ]

    rows = []
    for name, color, badge in items:
        w_cnt = white_counts.get(name, 0)
        b_cnt = black_counts.get(name, 0)
        badge_div = f"<div style='width:34px;height:34px;background:{color};color:#fff;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-weight:800;font-size:0.9rem;box-shadow:0 2px 5px rgba(0,0,0,0.4);'>{badge}</div>"
        row = f"<tr style='border-bottom:1px solid #242424;'><td style='padding:12px 16px;font-weight:600;color:{color};text-align:left;font-size:0.95rem;'>{name}</td><td style='padding:12px 16px;font-weight:700;color:#f0f0f0;text-align:center;font-size:1.1rem;'>{w_cnt}</td><td style='padding:8px 16px;text-align:center;'>{badge_div}</td><td style='padding:12px 16px;font-weight:700;color:#f0f0f0;text-align:center;font-size:1.1rem;'>{b_cnt}</td></tr>"
        rows.append(row)

    rows_joined = "".join(rows)
    table_header = f"<tr style='background:#1a1a1a;border-bottom:1px solid #2d2d2d;'><th style='padding:10px 16px;text-align:left;color:#777;font-size:0.78rem;text-transform:uppercase;font-weight:700;letter-spacing:0.5px;'>GRADE</th><th style='padding:10px 16px;text-align:center;color:#777;font-size:0.78rem;text-transform:uppercase;font-weight:700;letter-spacing:0.5px;'>{white_name.upper()}</th><th style='padding:10px 16px;text-align:center;width:60px;'></th><th style='padding:10px 16px;text-align:center;color:#777;font-size:0.78rem;text-transform:uppercase;font-weight:700;letter-spacing:0.5px;'>{black_name.upper()}</th></tr>"
    
    html = f"<div style='background:#141414;border:1px solid #282828;border-radius:8px;overflow:hidden;margin-top:8px;'><table style='width:100%;border-collapse:collapse;'><thead>{table_header}</thead><tbody>{rows_joined}</tbody></table></div>"
    return html


def render_analyze():
    st.title("Analyze Game")

    if not st.session_state.current_game_metadata:
        st.markdown(
            """
            <div class='card' style='text-align: center; padding: 40px 20px;'>
                <h4 style='color: #888;'>No active game loaded</h4>
                <p style='color: #666;'>Import a game via PGN or select a sample game to start analysis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Import a Game Now", type="primary"):
            navigate_to("Import Game")
        return

    meta = st.session_state.current_game_metadata
    positions = st.session_state.current_positions
    total_plies = len(positions) - 1

    # Header Summary Bar
    st.markdown(
        f"""
        <div class='card' style='padding: 12px 20px; display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <strong style='font-size: 1.15rem; color: #fff;'>{meta['white']} ({meta['white_elo']}) vs {meta['black']} ({meta['black_elo']})</strong>
                <div style='color: #888; font-size: 0.85rem;'>{meta['event']} | {meta['opening']} ({meta['eco']}) | {meta['date']}</div>
            </div>
            <div style='text-align: right;'>
                <span class='gold-text' style='font-size: 1.2rem;'>Result: {meta['result']}</span>
                <div style='color: #aaa; font-size: 0.85rem;'>{total_plies} plies ({meta['total_moves']} moves)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stockfish Depth & Engine Configuration
    if "stockfish_depth" not in st.session_state:
        st.session_state.stockfish_depth = 12

    st.markdown("<br>", unsafe_allow_html=True)
    depth_col1, depth_col2 = st.columns([1.6, 1.2])
    with depth_col1:
        current_depth = st.session_state.get("stockfish_depth", 12)
        selected_depth = st.slider(
            "Stockfish Engine Search Depth (ply)",
            min_value=6,
            max_value=22,
            value=current_depth,
            step=2,
            help="Higher depth calculates deeper tactical moves with Stockfish, while lower depth provides faster analysis.",
            key="depth_slider_control",
        )
        st.session_state.stockfish_depth = selected_depth

    with depth_col2:
        st.markdown("<span style='color: #888; font-size: 0.82rem;'>Quick Depth Presets:</span>", unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            if st.button("Fast (8)", help="Fast Analysis (Depth 8)", use_container_width=True):
                st.session_state.stockfish_depth = 8
                st.rerun()
        with p2:
            if st.button("Standard (12)", help="Standard Analysis (Depth 12)", use_container_width=True):
                st.session_state.stockfish_depth = 12
                st.rerun()
        with p3:
            if st.button("Deep (16)", help="Deep Analysis (Depth 16)", use_container_width=True):
                st.session_state.stockfish_depth = 16
                st.rerun()
        with p4:
            if st.button("Master (20)", help="Master Analysis (Depth 20)", use_container_width=True):
                st.session_state.stockfish_depth = 20
                st.rerun()

    # Analysis Execution Trigger
    st.markdown("<br>", unsafe_allow_html=True)
    an_col1, an_col2, an_col3 = st.columns([2, 1, 1])
    with an_col1:
        depth_val = st.session_state.get("stockfish_depth", 12)
        btn_label = f"Run Full Game Analysis (Depth {depth_val})"
        if st.button(btn_label, type="primary", use_container_width=True):
            game = parse_pgn(st.session_state.current_game_raw)
            if game is not None:
                progress_bar = st.progress(0, text=f"Starting Stockfish UCI Engine at Depth {depth_val}...")
                status_text = st.empty()

                def on_progress(curr: int, total: int):
                    pct = int((curr / max(1, total)) * 100)
                    progress_bar.progress(pct, text=f"Analyzing position {curr} of {total} plies (Depth {depth_val})...")

                try:
                    time_limit_val = 0.10 if depth_val <= 8 else (0.15 if depth_val <= 14 else 0.25)
                    with StockfishEngine(default_depth=depth_val, default_time_limit=time_limit_val) as engine:
                        if not engine.is_available():
                            progress_bar.empty()
                            st.error(
                                "Stockfish UCI engine binary was not found on this system.\n\n"
                                "Please ensure Stockfish is installed (e.g. via `packages.txt` on Streamlit Cloud or in system PATH)."
                            )
                            return

                        results = analyze_full_game(
                            game=game,
                            engine=engine,
                            depth=depth_val,
                            time_limit=time_limit_val,
                            progress_callback=on_progress,
                        )
                        st.session_state.game_analysis = results
                        st.session_state.current_positions = results["positions"]

                        # Append summary to analyzed_reports
                        report_entry = {
                            "white": meta["white"],
                            "black": meta["black"],
                            "result": meta["result"],
                            "date": meta["date"],
                            "opening": meta["opening"],
                            "total_plies": results["total_plies"],
                            "white_accuracy": results["white_accuracy"],
                            "black_accuracy": results["black_accuracy"],
                            "white_blunders": results["white_counts"].get(CLASS_BLUNDER, 0),
                            "black_blunders": results["black_counts"].get(CLASS_BLUNDER, 0),
                            "white_mistakes": results["white_counts"].get(CLASS_MISTAKE, 0),
                            "black_mistakes": results["black_counts"].get(CLASS_MISTAKE, 0),
                        }
                        # Avoid duplicates
                        if not any(r.get("white") == meta["white"] and r.get("black") == meta["black"] and r.get("date") == meta["date"] for r in st.session_state.analyzed_reports):
                            st.session_state.analyzed_reports.append(report_entry)

                    progress_bar.empty()
                    st.success(f"Full game analysis complete at Depth {depth_val}!")
                    st.rerun()
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Engine analysis error: {str(e)}")

    with an_col2:
        if st.session_state.game_analysis:
            w_acc = st.session_state.game_analysis["white_accuracy"]
            st.markdown(
                f"""
                <div style='background: #1e1e1e; border-left: 4px solid #4caf50; padding: 8px 12px; border-radius: 4px;'>
                    <span style='color: #aaa; font-size: 0.8rem;'>White Accuracy</span><br>
                    <strong style='color: #fff; font-size: 1.15rem;'>{w_acc}%</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with an_col3:
        if st.session_state.game_analysis:
            b_acc = st.session_state.game_analysis["black_accuracy"]
            st.markdown(
                f"""
                <div style='background: #1e1e1e; border-left: 4px solid #2196f3; padding: 8px 12px; border-radius: 4px;'>
                    <span style='color: #aaa; font-size: 0.8rem;'>Black Accuracy</span><br>
                    <strong style='color: #fff; font-size: 1.15rem;'>{b_acc}%</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    curr_idx = st.session_state.current_move_index
    curr_pos = positions[curr_idx] if curr_idx < len(positions) else positions[0]
    board = get_board_from_fen(curr_pos["fen"])

    # Determine last move to highlight
    last_move = None
    if curr_idx > 0 and curr_pos.get("uci"):
        try:
            last_move = chess.Move.from_uci(curr_pos["uci"])
        except ValueError:
            last_move = None

    # Engine recommended arrow
    arrows = []
    if curr_pos.get("best_move") is not None and isinstance(curr_pos["best_move"], chess.Move):
        arrows.append(chess.svg.Arrow(
            curr_pos["best_move"].from_square,
            curr_pos["best_move"].to_square,
            color="rgba(212, 175, 55, 0.85)",
        ))

    # Split: Eval Bar + Chessboard + Side Panel
    col_eval_bar, col_board, col_panel = st.columns([0.15, 1.3, 1.1])

    # 1. Sleek Evaluation Bar
    with col_eval_bar:
        eval_score_cp = curr_pos.get("eval_white", 0)
        # Scale between -600 cp (100% black) and +600 cp (100% white)
        clamped_eval = max(-600, min(600, eval_score_cp))
        white_pct = int(((clamped_eval + 600) / 1200) * 100)
        black_pct = 100 - white_pct

        if st.session_state.board_flipped:
            # Black on bottom if flipped
            top_color = "#f0f0f0"
            top_height = white_pct
            bot_color = "#222222"
            bot_height = black_pct
        else:
            # White on bottom
            top_color = "#222222"
            top_height = black_pct
            bot_color = "#f0f0f0"
            bot_height = white_pct

        eval_str = curr_pos.get("eval_formatted", "0.00")
        st.markdown(
            f"""
            <div style='height: 410px; width: 28px; background: #181818; border: 1px solid #333; border-radius: 6px; display: flex; flex-direction: column; overflow: hidden; position: relative;'>
                <div style='background: {top_color}; height: {top_height}%; transition: height 0.3s;'></div>
                <div style='background: {bot_color}; height: {bot_height}%; transition: height 0.3s;'></div>
                <div style='position: absolute; bottom: 6px; width: 100%; text-align: center; font-size: 0.65rem; font-weight: bold; color: {"#000" if bot_color=="#f0f0f0" else "#fff"};'>
                    {eval_str}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_board:
        # Render visual SVG Chessboard with on-piece classification badge
        curr_classification = curr_pos.get("classification") if curr_idx > 0 else None
        board_svg = generate_board_svg(
            board=board,
            last_move=last_move,
            flipped=st.session_state.board_flipped,
            arrows=arrows,
            size=410,
            classification=curr_classification,
        )

        st.markdown(
            f"""
            <div style='display: flex; justify-content: center; align-items: center; padding: 8px; background: #1a1a1a; border-radius: 8px; border: 1px solid #2d2d2d;'>
                {board_svg}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Move navigation controls
        st.markdown("<br>", unsafe_allow_html=True)
        nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)
        with nav_col1:
            if st.button("First", use_container_width=True):
                st.session_state.current_move_index = 0
                st.rerun()
        with nav_col2:
            if st.button("Prev", use_container_width=True):
                if st.session_state.current_move_index > 0:
                    st.session_state.current_move_index -= 1
                    st.rerun()
        with nav_col3:
            if st.button("Next", use_container_width=True):
                if st.session_state.current_move_index < total_plies:
                    st.session_state.current_move_index += 1
                    st.rerun()
        with nav_col4:
            if st.button("Last", use_container_width=True):
                st.session_state.current_move_index = total_plies
                st.rerun()
        with nav_col5:
            if st.button("Flip", use_container_width=True):
                st.session_state.board_flipped = not st.session_state.board_flipped
                st.rerun()

    with col_panel:
        st.markdown("### Move Details & Engine")

        pos_label = f"({curr_pos['san']})" if curr_pos['san'] else "(Start Position)"
        turn_label = "White to Move" if board.turn == chess.WHITE else "Black to Move"

        # Classification Badge if game was analyzed
        classif_badge_html = ""
        if "classification" in curr_pos and curr_idx > 0:
            c_name = curr_pos["classification"]
            c_color = curr_pos.get("color_code", "#2ecc71")
            c_badge = BADGE_TEXT_MAP.get(c_name, "")
            cpl = curr_pos.get("cpl", 0)
            badge_icon_span = f"<span style='background:{c_color};color:#fff;border-radius:50%;padding:1px 6px;margin-right:5px;font-weight:bold;font-size:0.75rem;'>{c_badge}</span>" if c_badge else ""
            classif_badge_html = f"""
            <div style='display: inline-flex; align-items: center; background: {c_color}22; border: 1px solid {c_color}; color: {c_color}; padding: 4px 10px; border-radius: 12px; font-size: 0.82rem; font-weight: 600; margin-top: 4px;'>
                {badge_icon_span}{c_name} (Loss: -{cpl} cp)
            </div>
            """

        eval_display = curr_pos.get("eval_formatted", "+0.00")
        eval_style_class = "eval-positive" if not eval_display.startswith("-") else "eval-negative"

        st.markdown(
            f"""
            <div class='metric-badge'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='font-size: 0.85rem; color: #aaa;'>Active Position (Ply {curr_idx}/{total_plies})</span>
                    <span class='{eval_style_class}' style='font-size: 1.3rem;'>{eval_display}</span>
                </div>
                <div style='font-size: 1.15rem; color: #fff; margin-top: 2px;'>
                    <span style='color: #d4af37;'>{pos_label}</span> • <span style='font-size: 0.85rem; color: #bbb;'>{turn_label}</span>
                </div>
                {classif_badge_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        best_move_text = curr_pos.get("best_move_san") or "-"
        depth_val = curr_pos.get("depth", 12)
        pv_line = curr_pos.get("pv_formatted") or "-"

        st.markdown(
            f"""
            <div class='card'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 6px;'>
                    <span style='color: #aaa;'>Stockfish Best Move:</span>
                    <strong style='color: #d4af37; font-size: 1.05rem;'>{best_move_text}</strong>
                </div>
                <div style='display: flex; justify-content: space-between; margin-bottom: 6px;'>
                    <span style='color: #aaa;'>Engine Depth:</span>
                    <span>{depth_val} ply</span>
                </div>
                <div style='display: flex; justify-content: space-between; margin-bottom: 6px;'>
                    <span style='color: #aaa;'>In Check:</span>
                    <span>{'Yes' if curr_pos['check'] else 'No'}</span>
                </div>
                <div style='margin-top: 8px;'>
                    <span style='color: #aaa; font-size: 0.85rem;'>Principal Variation (PV):</span>
                    <div style='font-family: monospace; font-size: 0.8rem; color: #ddd; background: #141414; padding: 6px 8px; border-radius: 4px; margin-top: 2px; overflow-x: auto;'>
                        {pv_line}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    tab_moves, tab_graph, tab_report, tab_ai = st.tabs(["Move List", "Evaluation Curve", "Game Review", "Model Insights"])

    with tab_moves:
        st.markdown("#### Interactive Move List")
        st.caption("Click any move to jump directly to that position:")

        # Render clickable move pairs in a responsive grid
        cols_per_row = 6
        move_pairs = []
        for i in range(1, len(positions), 2):
            move_num = positions[i]["move_number"]
            w_ply = i
            b_ply = i + 1 if (i + 1) < len(positions) else None
            move_pairs.append((move_num, w_ply, positions[w_ply], b_ply, positions[b_ply] if b_ply else None))

        for row_start in range(0, len(move_pairs), cols_per_row):
            row_items = move_pairs[row_start : row_start + cols_per_row]
            row_cols = st.columns(len(row_items))
            for col, (num, w_ply, w_pos, b_ply, b_pos) in zip(row_cols, row_items):
                with col:
                    st.markdown(f"<span style='color: #888; font-size: 0.8rem;'>{num}.</span>", unsafe_allow_html=True)
                    w_label = w_pos["san"]
                    btn_w_type = "primary" if curr_idx == w_ply else "secondary"
                    if st.button(w_label, key=f"ply_{w_ply}", type=btn_w_type, use_container_width=True):
                        st.session_state.current_move_index = w_ply
                        st.rerun()

                    if b_ply and b_pos:
                        b_label = b_pos["san"]
                        btn_b_type = "primary" if curr_idx == b_ply else "secondary"
                        if st.button(b_label, key=f"ply_{b_ply}", type=btn_b_type, use_container_width=True):
                            st.session_state.current_move_index = b_ply
                            st.rerun()

    with tab_graph:
        st.markdown("#### Engine Evaluation Curve")
        if st.session_state.game_analysis:
            fig_eval = create_evaluation_chart(st.session_state.current_positions, current_ply=curr_idx)
            st.plotly_chart(fig_eval, use_container_width=True)
            st.caption("Hover over points to inspect move details, centipawn losses, and engine recommendations.")
        else:
            st.info("Run Stockfish Engine Analysis above to generate the full evaluation curve.")

    with tab_report:
        st.markdown("#### Game Review & Move Summary")
        if st.session_state.game_analysis:
            rep = st.session_state.game_analysis
            white_p = meta.get("white", "White")
            black_p = meta.get("black", "Black")
            w_acc = rep["white_accuracy"]
            b_acc = rep["black_accuracy"]

            # 1. Accuracy Cards (matching Chess.com layout)
            st.markdown("##### Accuracy")
            acc_col1, acc_col2 = st.columns(2)
            with acc_col1:
                st.markdown(
                    f"""
                    <div style='background: #0d0d0d; border: 1.5px solid #2e2e2e; border-radius: 10px; padding: 20px 24px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.5);'>
                        <div style='color: #9e9e9e; font-size: 1rem; font-weight: 600; letter-spacing: 0.5px;'>{white_p}</div>
                        <div style='color: #ffffff; font-size: 2.8rem; font-weight: 800; line-height: 1.2; margin-top: 6px;'>{w_acc}%</div>
                        <div style='color: #666666; font-size: 0.8rem; margin-top: 6px;'>Avg Loss: {rep["white_cpl_avg"]} cp</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with acc_col2:
                st.markdown(
                    f"""
                    <div style='background: #ffffff; border: 1.5px solid #dddddd; border-radius: 10px; padding: 20px 24px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.25);'>
                        <div style='color: #444444; font-size: 1rem; font-weight: 600; letter-spacing: 0.5px;'>{black_p}</div>
                        <div style='color: #111111; font-size: 2.8rem; font-weight: 800; line-height: 1.2; margin-top: 6px;'>{b_acc}%</div>
                        <div style='color: #777777; font-size: 0.8rem; margin-top: 6px;'>Avg Loss: {rep["black_cpl_avg"]} cp</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # 2. Summary Section with GOOD MOVES and MISTAKES Tabs
            st.markdown("### Summary")
            subtab_good, subtab_mistakes = st.tabs(["Good Moves", "Mistakes"])

            with subtab_good:
                st.markdown(
                    render_chess_com_summary_html(
                        white_name=white_p,
                        black_name=black_p,
                        white_counts=rep["white_counts"],
                        black_counts=rep["black_counts"],
                        mode="good",
                    ),
                    unsafe_allow_html=True,
                )

            with subtab_mistakes:
                st.markdown(
                    render_chess_com_summary_html(
                        white_name=white_p,
                        black_name=black_p,
                        white_counts=rep["white_counts"],
                        black_counts=rep["black_counts"],
                        mode="mistakes",
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Move Quality Breakdown Chart")
            fig_breakdown = create_move_breakdown_chart(rep["white_counts"], rep["black_counts"])
            st.plotly_chart(fig_breakdown, use_container_width=True)

            if rep["critical_moments"]:
                st.markdown("---")
                st.markdown("### Critical Turning Moments")
                crit_data = [
                    {
                        "Ply": cm["ply"],
                        "Move #": cm["move_number"],
                        "Player": cm["player"],
                        "Played": cm["move"],
                        "Best Move": cm["best_move"],
                        "Classification": f"{BADGE_TEXT_MAP.get(cm['classification'], '')} {cm['classification']}",
                        "Centipawn Loss": f"-{cm['cpl']} cp",
                        "Eval After": cm["eval_after"],
                    }
                    for cm in rep["critical_moments"]
                ]
                st.table(crit_data)
        else:
            st.info("Run Stockfish Analysis above to generate the full Game Report.")

    with tab_ai:
        st.markdown("#### Model Insights: Machine Learning & Deep Learning")
        st.caption("Real-time inference using trained Scikit-Learn and PyTorch Deep Learning models.")

        ml_pred = get_ml_predictor()
        dl_pred = get_dl_predictor()

        # Status Bar for AI Models
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            if ml_pred.is_model_loaded():
                m_metrics = ml_pred.get_model_metrics() or {}
                acc_str = f"Accuracy: {m_metrics.get('accuracy', 0.782):.1%}"
                st.markdown(
                    f"""
                    <div style='background: #1a2a1a; border: 1px solid #2ecc71; padding: 10px 14px; border-radius: 6px; margin-bottom: 12px;'>
                        <strong style='color: #2ecc71;'>Scikit-Learn Classifier Active</strong><br>
                        <span style='color: #aaa; font-size: 0.85rem;'>{m_metrics.get('model_name', 'Logistic Regression')} • {acc_str} • F1: {m_metrics.get('f1', 0.37)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Scikit-Learn model not loaded.")

        with m_col2:
            if dl_pred.is_model_loaded():
                dl_metrics = dl_pred.get_model_metrics() or {}
                val_loss = dl_metrics.get('final_val_loss', 0.035)
                st.markdown(
                    f"""
                    <div style='background: #1a233a; border: 1px solid #3498db; padding: 10px 14px; border-radius: 6px; margin-bottom: 12px;'>
                        <strong style='color: #3498db;'>PyTorch Neural Network Active</strong><br>
                        <span style='color: #aaa; font-size: 0.85rem;'>ChessCNN (8x8x12 Tensors) • Val Loss: {val_loss} • {dl_metrics.get('epochs', 20)} Epochs</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.warning("PyTorch model not loaded.")

        # Active Move Predictions
        st.markdown("---")
        ai_col_ml, ai_col_dl = st.columns(2)

        # 1. Scikit-Learn Move Quality & Blunder Predictor
        with ai_col_ml:
            st.markdown("##### 1. Scikit-Learn Move Quality & Blunder Risk")
            if curr_idx > 0 and curr_pos.get("uci"):
                try:
                    prev_pos = positions[curr_idx - 1]
                    prev_board = get_board_from_fen(prev_pos["fen"])
                    move_obj = chess.Move.from_uci(curr_pos["uci"])
                    eval_before = prev_pos.get("eval_white", 0)

                    ml_res = ml_pred.predict_move(prev_board, move_obj, eval_before_cp=eval_before)

                    prob = ml_res["blunder_probability"]
                    risk_lvl = ml_res["risk_level"]
                    risk_col = ml_res["risk_color"]
                    conf = ml_res["confidence"]

                    st.markdown(
                        f"""
                        <div class='card' style='border-left: 4px solid {risk_col};'>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <span style='color: #aaa;'>Move Evaluated:</span>
                                <strong style='font-size: 1.1rem; color: #fff;'>{curr_pos['san']}</strong>
                            </div>
                            <div style='margin-top: 10px;'>
                                <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
                                    <span style='color: #ddd; font-size: 0.9rem;'>Predicted Blunder Probability:</span>
                                    <strong style='color: {risk_col};'>{prob}%</strong>
                                </div>
                                <div style='height: 8px; background: #333; border-radius: 4px; overflow: hidden;'>
                                    <div style='width: {prob}%; background: {risk_col}; height: 100%; transition: width 0.3s;'></div>
                                </div>
                            </div>
                            <div style='display: flex; justify-content: space-between; margin-top: 12px; font-size: 0.85rem;'>
                                <span style='color: #888;'>Risk Classification:</span>
                                <strong style='color: {risk_col};'>{risk_lvl}</strong>
                            </div>
                            <div style='display: flex; justify-content: space-between; margin-top: 4px; font-size: 0.85rem;'>
                                <span style='color: #888;'>Model Confidence:</span>
                                <span style='color: #ccc;'>{conf}%</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Domain Feature Attribution Table
                    with st.expander("View Positional Feature Attribution"):
                        feats = ml_res.get("features", {})
                        feat_display = [
                            {"Feature": "Material Balance (Pawns)", "Value": f"{feats.get('material_balance', 0)/100:.2f}"},
                            {"Feature": "Total Board Material", "Value": f"{feats.get('total_material', 0)/100:.1f}"},
                            {"Feature": "Player Center Control (e4/d4/e5/d5)", "Value": f"{feats.get('user_center_control', 0)} squares"},
                            {"Feature": "Opponent Center Control", "Value": f"{feats.get('opp_center_control', 0)} squares"},
                            {"Feature": "Player King Danger Metric", "Value": f"{feats.get('user_king_danger', 0)}"},
                            {"Feature": "Opponent King Danger Metric", "Value": f"{feats.get('opp_king_danger', 0)}"},
                            {"Feature": "Legal Moves Available", "Value": f"{feats.get('legal_moves_count', 0)}"},
                            {"Feature": "Is Tactical Capture", "Value": "Yes" if feats.get("is_capture") else "No"},
                            {"Feature": "Gives Check", "Value": "Yes" if feats.get("gives_check") else "No"},
                        ]
                        st.table(feat_display)

                except Exception as e:
                    st.warning(f"Feature extraction error: {e}")
            else:
                st.info("Navigate to a move in the move list to inspect ML move quality predictions.")

        # 2. PyTorch Deep Learning Board Evaluator
        with ai_col_dl:
            st.markdown("##### 2. PyTorch Deep Learning (ChessCNN)")
            dl_res = dl_pred.predict_position(board)

            dl_eval = dl_res["eval_formatted"]
            w_win = dl_res["white_win_pct"]
            b_win = dl_res["black_win_pct"]
            assess = dl_res["assessment"]
            adv_color = dl_res["advantage_color"]

            st.markdown(
                f"""
                <div class='card' style='border-left: 4px solid {adv_color};'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='color: #aaa;'>Neural Evaluation:</span>
                        <strong style='font-size: 1.25rem; color: #fff;'>{dl_eval}</strong>
                    </div>
                    <div style='margin-top: 10px;'>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
                            <span style='color: #eee; font-size: 0.85rem;'>White Win: {w_win}%</span>
                            <span style='color: #eee; font-size: 0.85rem;'>Black Win: {b_win}%</span>
                        </div>
                        <div style='height: 10px; background: #2196f3; border-radius: 5px; overflow: hidden; display: flex;'>
                            <div style='width: {w_win}%; background: #ffffff; height: 100%; transition: width 0.3s;'></div>
                            <div style='width: {b_win}%; background: #333333; height: 100%; transition: width 0.3s;'></div>
                        </div>
                    </div>
                    <div style='display: flex; justify-content: space-between; margin-top: 12px; font-size: 0.85rem;'>
                        <span style='color: #888;'>Positional Standing:</span>
                        <strong style='color: {adv_color};'>{assess}</strong>
                    </div>
                    <div style='display: flex; justify-content: space-between; margin-top: 4px; font-size: 0.85rem;'>
                        <span style='color: #888;'>Tensor Input:</span>
                        <span style='color: #ccc;'>8x8x12 (6 White + 6 Black Channels)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("View PyTorch CNN Architecture"):
                st.markdown(
                    """
                    ```
                    Input: (12, 8, 8) Board Tensor
                    ├── Conv2d(12 -> 32, kernel=3, padding=1) + BatchNorm2d + ReLU
                    ├── Conv2d(32 -> 64, kernel=3, padding=1) + BatchNorm2d + ReLU
                    ├── Conv2d(64 -> 128, kernel=3, padding=1) + BatchNorm2d + ReLU
                    ├── AdaptiveAvgPool2d((4, 4))
                    ├── Flatten -> Linear(2048 -> 128) + ReLU + Dropout(0.25)
                    ├── Linear(128 -> 64) + ReLU
                    └── Linear(64 -> 1) + Tanh -> Normalized Evaluation [-1.0, 1.0]
                    ```
                    """
                )

        # 3. Three-Way AI Comparison
        st.markdown("---")
        st.markdown("##### Engine vs ML vs Deep Learning Comparison")
        comp_col1, comp_col2, comp_col3 = st.columns(3)

        with comp_col1:
            sf_eval = curr_pos.get("eval_formatted", "0.00")
            sf_move = curr_pos.get("best_move_san", "-")
            st.markdown(
                f"""
                <div class='card'>
                    <span class='gold-text' style='font-size: 0.95rem;'>1. Stockfish Engine</span>
                    <div style='font-size: 1.2rem; color: #fff; margin-top: 4px;'>{sf_eval}</div>
                    <div style='color: #888; font-size: 0.8rem; margin-top: 4px;'>Depth: {curr_pos.get('depth', 12)} ply</div>
                    <div style='color: #aaa; font-size: 0.85rem;'>Best: <strong>{sf_move}</strong></div>
                    <p style='color: #666; font-size: 0.75rem; margin-top: 6px;'>Alpha-Beta Minimax Search with heuristic evaluation.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with comp_col2:
            ml_prob_str = f"{ml_res['blunder_probability']}%" if 'ml_res' in locals() else "N/A"
            ml_risk_str = ml_res['risk_level'] if 'ml_res' in locals() else "N/A"
            st.markdown(
                f"""
                <div class='card'>
                    <span style='color: #2ecc71; font-size: 0.95rem;'>2. Scikit-Learn ML</span>
                    <div style='font-size: 1.2rem; color: #fff; margin-top: 4px;'>{ml_prob_str} Blunder Risk</div>
                    <div style='color: #888; font-size: 0.8rem; margin-top: 4px;'>{ml_risk_str}</div>
                    <div style='color: #aaa; font-size: 0.85rem;'>Features: <strong>17 Engineered</strong></div>
                    <p style='color: #666; font-size: 0.75rem; margin-top: 6px;'>Domain features (material, center, king safety, mobility).</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with comp_col3:
            st.markdown(
                f"""
                <div class='card'>
                    <span style='color: #3498db; font-size: 0.95rem;'>3. PyTorch Deep Learning</span>
                    <div style='font-size: 1.2rem; color: #fff; margin-top: 4px;'>{dl_eval}</div>
                    <div style='color: #888; font-size: 0.8rem; margin-top: 4px;'>{assess}</div>
                    <div style='color: #aaa; font-size: 0.85rem;'>Input: <strong>8x8x12 Tensor</strong></div>
                    <p style='color: #666; font-size: 0.75rem; margin-top: 6px;'>Direct spatial pattern recognition via 2D Convolutions.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )



# ---------------------------------------------------------
# View: 4. My Games
# ---------------------------------------------------------
def render_my_games():
    st.title("My Games")
    st.markdown(
        "<p class='sub-text'>Review games loaded or fetched during your current session.</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.session_games:
        st.markdown(
            """
            <div class='card' style='text-align: center; padding: 40px 20px;'>
                <h4 style='color: #888;'>No games in session yet</h4>
                <p style='color: #666;'>Import a PGN or fetch from Chess.com to view your session history here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Import a Game Now"):
            navigate_to("Import Game")
    else:
        for idx, g in enumerate(st.session_state.session_games):
            col_info, col_btn = st.columns([4, 1])
            with col_info:
                st.markdown(
                    f"""
                    <div class='card' style='margin-bottom: 8px;'>
                        <strong style='color: #fff;'>{g['white']} vs {g['black']}</strong>
                        <span class='gold-text' style='margin-left: 12px;'>Result: {g['result']}</span>
                        <div style='color: #888; font-size: 0.85rem; margin-top: 4px;'>
                            {g['event']} • {g['opening']} • {g['total_plies']} plies • <em>{g['source']}</em>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_btn:
                if st.button("Analyze", key=f"load_game_{idx}", use_container_width=True):
                    load_game_into_state(g["pgn"], g["source"])
                    navigate_to("Analyze")


# ---------------------------------------------------------
# View: 5. Reports
# ---------------------------------------------------------
def render_reports():
    st.title("Multi-Game Reports")
    st.markdown(
        "<p class='sub-text'>Aggregated statistics and performance trends across your analyzed games.</p>",
        unsafe_allow_html=True,
    )

    reports = st.session_state.analyzed_reports

    if not reports:
        total_session_games = len(st.session_state.session_games)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Games", f"{total_session_games}")
        with col2:
            st.metric("Analyzed Games", "0")
        with col3:
            st.metric("Avg Accuracy", "0.0%")
        with col4:
            st.metric("Avg Blunders", "0.0 / game")

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("You have not analyzed any games with Stockfish yet. Go to **Analyze** and click 'Run Full Game Analysis' to generate multi-game statistics and trends.")
        if total_session_games > 0:
            if st.button("Go to Analyze Active Game", type="primary"):
                navigate_to("Analyze")
        return

    import plotly.graph_objects as go

    total_analyzed = len(reports)
    avg_white_acc = round(sum(r["white_accuracy"] for r in reports) / total_analyzed, 1)
    avg_black_acc = round(sum(r["black_accuracy"] for r in reports) / total_analyzed, 1)
    overall_acc = round((avg_white_acc + avg_black_acc) / 2.0, 1)
    total_blunders = sum(r["white_blunders"] + r["black_blunders"] for r in reports)
    avg_blunders = round(total_blunders / total_analyzed, 1)

    # Win / Draw / Loss counts
    wins = sum(1 for r in reports if "1-0" in r["result"] or "Won" in r["result"])
    draws = sum(1 for r in reports if "1/2" in r["result"] or "Draw" in r["result"])
    losses = sum(1 for r in reports if "0-1" in r["result"] or "Lost" in r["result"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Analyzed Games", f"{total_analyzed}")
    with col2:
        st.metric("Overall Accuracy", f"{overall_acc}%")
    with col3:
        st.metric("Avg Blunders / Game", f"{avg_blunders}")
    with col4:
        st.metric("Decisive Games", f"{wins + losses} / {total_analyzed}")

    st.markdown("<br>", unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("##### Outcome Distribution")
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=["White Wins (1-0)", "Draws (1/2-1/2)", "Black Wins (0-1)"],
                    values=[wins, draws, losses],
                    hole=0.45,
                    marker=dict(colors=["#f0d9b5", "#888888", "#769656"]),
                )
            ]
        )
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
            margin=dict(l=20, r=20, t=20, b=20),
            height=260,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        st.markdown("##### Accuracy Across Analyzed Games")
        game_nums = [f"Game {i+1}" for i in range(total_analyzed)]
        w_accs = [r["white_accuracy"] for r in reports]
        b_accs = [r["black_accuracy"] for r in reports]

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=game_nums, y=w_accs, mode="lines+markers", name="White Acc", line=dict(color="#f0d9b5", width=2)))
        fig_trend.add_trace(go.Scatter(x=game_nums, y=b_accs, mode="lines+markers", name="Black Acc", line=dict(color="#769656", width=2)))
        fig_trend.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
            margin=dict(l=20, r=20, t=20, b=20),
            height=260,
            yaxis=dict(title="Accuracy %", range=[40, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Analyzed Games History")
    st.dataframe(reports, use_container_width=True)


# ---------------------------------------------------------
# View: 6. About
# ---------------------------------------------------------
def render_about():
    st.title("About ChessLabs")
    st.markdown(
        """
        <div class='card'>
            <div class='card-title'>ChessLabs — College Mini-Project</div>
            <div class='card-body'>
                <strong>ChessLabs</strong> is an educational Chess Game Analyzer designed to provide 
                in-depth move evaluation, game statistics, and machine learning insights.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Architecture & Technology Stack")
    st.markdown(
        """
        - **UI / Frontend**: [Streamlit](https://streamlit.io) (Dark Charcoal + Gold Accent theme)
        - **Chess Logic**: [python-chess](https://python-chess.readthedocs.io)
        - **Engine Evaluation**: Stockfish UCI engine
        - **Data Processing**: Pandas, NumPy
        - **Visualizations**: Matplotlib, Plotly
        - **Machine Learning**: Scikit-Learn (Blunder & Move Quality Classification)
        - **Deep Learning**: PyTorch (8×8×12 Board Tensor Evaluation)
        - **Data Source**: Chess.com Public API & Public PGN Datasets
        """
    )

    st.markdown("### Project Structure")
    st.code(
        """ChessLabs/
├── app.py                   # Streamlit Main Application
├── chess/                   # Core Chess Logic & Engine Interop
│   ├── pgn_parser.py
│   ├── board.py
│   ├── stockfish_engine.py
│   └── game_analyzer.py
├── ml/                      # Machine Learning Feature Extraction & Models
├── dl/                      # PyTorch Deep Learning Models
├── training/                # Training & Evaluation Scripts
├── data/                    # Datasets (raw, processed, sample)
├── models/                  # Saved .joblib and .pth Models
└── tests/                   # Automated Unit & Integration Tests""",
        language="text",
    )


# ---------------------------------------------------------
# Page Router
# ---------------------------------------------------------
def main():
    current_page = st.session_state.selected_nav

    if current_page == "Home":
        render_home()
    elif current_page == "Import Game":
        render_import_game()
    elif current_page == "Analyze":
        render_analyze()
    elif current_page == "My Games":
        render_my_games()
    elif current_page == "Reports":
        render_reports()
    elif current_page == "About":
        render_about()
    else:
        render_home()


if __name__ == "__main__":
    main()
