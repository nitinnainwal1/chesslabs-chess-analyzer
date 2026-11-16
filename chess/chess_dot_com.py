"""ChessLabs - Chess.com Public API Client.

Interacts with the official Chess.com Published Data API to fetch
public game archives and format recent matches for analysis.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import requests


# Chess.com Public API requires a custom descriptive User-Agent
USER_AGENT = "ChessLabs-App/1.0 (Chess Analyzer Mini-Project; dev@chesslabs.local)"
BASE_API_URL = "https://api.chess.com/pub/player"


class ChessDotComError(Exception):
    """Base exception for Chess.com API interactions."""
    pass


def fetch_user_archives(username: str, timeout: float = 8.0) -> List[str]:
    """Fetch the list of monthly archive URLs for a given Chess.com username.

    Args:
        username: Chess.com username (case-insensitive).
        timeout: Request timeout in seconds.

    Returns:
        List of archive URLs sorted chronologically (oldest to newest).

    Raises:
        ChessDotComError: If user is not found, rate limited, or connection fails.
    """
    clean_user = username.strip().lower()
    if not clean_user:
        raise ChessDotComError("Username cannot be empty.")

    url = f"{BASE_API_URL}/{clean_user}/games/archives"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 404:
            raise ChessDotComError(f"User '{username}' was not found on Chess.com.")
        elif response.status_code == 429:
            raise ChessDotComError("Chess.com API rate limit exceeded. Please wait a moment and try again.")
        elif response.status_code != 200:
            raise ChessDotComError(f"Chess.com API error (HTTP {response.status_code}).")

        data = response.json()
        return data.get("archives", [])
    except requests.exceptions.RequestException as e:
        raise ChessDotComError(f"Network error while connecting to Chess.com: {str(e)}") from e


def fetch_archive_games(archive_url: str, timeout: float = 8.0) -> List[Dict[str, Any]]:
    """Fetch all games from a specific monthly archive URL.

    Args:
        archive_url: URL to a monthly game archive.
        timeout: Request timeout in seconds.

    Returns:
        List of raw game dictionaries.
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    try:
        response = requests.get(archive_url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return []
        data = response.json()
        return data.get("games", [])
    except Exception:
        return []


def fetch_recent_games(
    username: str,
    limit: int = 20,
    time_class: Optional[str] = None,
    timeout: float = 8.0,
) -> List[Dict[str, Any]]:
    """Fetch the most recent games for a Chess.com user with optional time-class filtering.

    Args:
        username: Chess.com username.
        limit: Maximum number of recent games to return.
        time_class: Optional filter: 'rapid', 'blitz', 'bullet', 'daily', or None for all.
        timeout: Request timeout in seconds.

    Returns:
        List of formatted game dictionaries ordered newest first.
    """
    archives = fetch_user_archives(username, timeout=timeout)
    if not archives:
        return []

    # Iterate through archives starting from most recent month backwards
    collected_games: List[Dict[str, Any]] = []
    target_user_lower = username.strip().lower()

    for archive_url in reversed(archives):
        month_games = fetch_archive_games(archive_url, timeout=timeout)
        # Sort month games newest first based on end_time
        for g in reversed(month_games):
            g_time_class = g.get("time_class", "").lower()
            if time_class and time_class.lower() != "all":
                if g_time_class != time_class.lower():
                    continue

            # Must contain PGN
            pgn = g.get("pgn", "")
            if not pgn.strip():
                continue

            formatted_entry = format_game_entry(g, target_user_lower)
            collected_games.append(formatted_entry)

            if len(collected_games) >= limit:
                return collected_games

    return collected_games


def format_game_entry(game_data: Dict[str, Any], target_user_lower: str) -> Dict[str, Any]:
    """Extract and format user-friendly game summary information.

    Args:
        game_data: Raw JSON game object from Chess.com API.
        target_user_lower: Target username in lowercase.

    Returns:
        Standardized game metadata dictionary.
    """
    white = game_data.get("white", {})
    black = game_data.get("black", {})

    w_username = white.get("username", "Unknown")
    b_username = black.get("username", "Unknown")
    w_rating = white.get("rating", "?")
    b_rating = black.get("rating", "?")

    is_user_white = (w_username.lower() == target_user_lower)
    user_color = "White" if is_user_white else "Black"
    user_rating = w_rating if is_user_white else b_rating
    opponent_name = b_username if is_user_white else w_username
    opponent_rating = b_rating if is_user_white else w_rating

    # Result determination
    w_result = white.get("result", "")
    b_result = black.get("result", "")

    if is_user_white:
        user_result = "Won 🏆" if w_result == "win" else ("Draw 🤝" if w_result in ("agreed", "repetition", "stalemate", "timevsinsufficient", "insufficient") else "Lost ❌")
    else:
        user_result = "Won 🏆" if b_result == "win" else ("Draw 🤝" if b_result in ("agreed", "repetition", "stalemate", "timevsinsufficient", "insufficient") else "Lost ❌")

    # Date formatting
    end_time_stamp = game_data.get("end_time")
    if end_time_stamp:
        from datetime import timezone
        date_str = datetime.fromtimestamp(end_time_stamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    else:
        date_str = "Recent"

    time_class = game_data.get("time_class", "Standard").capitalize()
    time_control = game_data.get("time_control", "-")

    return {
        "url": game_data.get("url", ""),
        "user_color": user_color,
        "user_rating": user_rating,
        "opponent": opponent_name,
        "opponent_rating": opponent_rating,
        "result": user_result,
        "time_class": time_class,
        "time_control": time_control,
        "date": date_str,
        "white": w_username,
        "black": b_username,
        "pgn": game_data.get("pgn", ""),
    }
