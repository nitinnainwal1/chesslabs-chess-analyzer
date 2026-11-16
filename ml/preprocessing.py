"""ChessLabs - ML Preprocessing & Leakage-Free Splitting.

Splits positions at the game level using GroupShuffleSplit so that positions
from the same game do not appear in both train and test sets.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from ml.features import FEATURE_COLUMNS


def split_and_scale_data(
    df: pd.DataFrame,
    target_column: str = "is_blunder",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """Split dataset by game_id to avoid data leakage, and scale features.

    Args:
        df: DataFrame containing FEATURE_COLUMNS, target_column, and 'game_id'.
        target_column: Column name of target label ('is_blunder' or 'quality_class').
        test_size: Fraction of games allocated to test set.
        random_state: Random seed for reproducibility.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    # Select available feature columns
    features = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[features].values
    y = df[target_column].values

    if "game_id" in df.columns and len(df["game_id"].unique()) > 1:
        groups = df["game_id"].values
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        # Fallback if single game
        split_point = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features
