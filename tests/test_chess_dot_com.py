"""Unit tests for chess/chess_dot_com.py."""

from unittest.mock import patch, MagicMock
import pytest
import requests
from chess.chess_dot_com import (
    fetch_user_archives,
    fetch_recent_games,
    format_game_entry,
    ChessDotComError,
)


def test_fetch_user_archives_success():
    """Test successful monthly archive retrieval."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "archives": [
            "https://api.chess.com/pub/player/hikaru/games/2024/01",
            "https://api.chess.com/pub/player/hikaru/games/2024/02",
        ]
    }

    with patch("requests.get", return_value=mock_resp):
        archives = fetch_user_archives("hikaru")
        assert len(archives) == 2
        assert "2024/02" in archives[1]


def test_fetch_user_archives_not_found():
    """Test 404 error when user does not exist."""
    mock_resp = MagicMock()
    mock_resp.status_code = 404

    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(ChessDotComError) as exc_info:
            fetch_user_archives("non_existent_chess_player_9999")
        assert "not found" in str(exc_info.value).lower()


def test_format_game_entry_white_win():
    """Test formatting raw game object where target user won as White."""
    raw_game = {
        "url": "https://www.chess.com/game/live/12345",
        "pgn": "1. e4 e5 2. Nf3 1-0",
        "time_control": "180+2",
        "time_class": "blitz",
        "end_time": 1700000000,
        "white": {"username": "MagnusCarlsen", "rating": 2880, "result": "win"},
        "black": {"username": "OpponentPlayer", "rating": 2800, "result": "resigned"},
    }

    entry = format_game_entry(raw_game, "magnuscarlsen")
    assert entry["user_color"] == "White"
    assert entry["user_rating"] == 2880
    assert entry["opponent"] == "OpponentPlayer"
    assert entry["opponent_rating"] == 2800
    assert "Won" in entry["result"]
    assert entry["time_class"] == "Blitz"


def test_fetch_recent_games_with_filter():
    """Test fetching recent games filtered by time control."""
    mock_archives_resp = MagicMock()
    mock_archives_resp.status_code = 200
    mock_archives_resp.json.return_value = {"archives": ["https://api.chess.com/pub/player/testuser/games/2024/01"]}

    mock_games_resp = MagicMock()
    mock_games_resp.status_code = 200
    mock_games_resp.json.return_value = {
        "games": [
            {
                "url": "https://chess.com/1",
                "pgn": "1. e4 e5 1-0",
                "time_class": "rapid",
                "white": {"username": "testuser", "rating": 1500, "result": "win"},
                "black": {"username": "other", "rating": 1450, "result": "checkmated"},
            },
            {
                "url": "https://chess.com/2",
                "pgn": "1. d4 d5 0-1",
                "time_class": "blitz",
                "white": {"username": "other", "rating": 1600, "result": "win"},
                "black": {"username": "testuser", "rating": 1550, "result": "resigned"},
            },
        ]
    }

    def side_effect(url, **kwargs):
        if "archives" in url:
            return mock_archives_resp
        return mock_games_resp

    with patch("requests.get", side_effect=side_effect):
        # Filter only rapid
        rapid_games = fetch_recent_games("testuser", time_class="rapid")
        assert len(rapid_games) == 1
        assert rapid_games[0]["time_class"] == "Rapid"

        # All games
        all_games = fetch_recent_games("testuser", time_class="all")
        assert len(all_games) == 2
