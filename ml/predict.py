"""ChessLabs - ML Inference Engine.

Loads the trained move quality model (Joblib) and predicts blunder risk
and positional complexity for arbitrary chess moves in real time.
"""

import os
from typing import Any, Dict, List, Optional
import chess
import joblib
import numpy as np

from ml.features import extract_features, FEATURE_COLUMNS


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "move_quality_model.joblib",
)


class MLMovePredictor:
    """Inference wrapper for trained Scikit-Learn Move Quality & Blunder model."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self._bundle: Optional[Dict[str, Any]] = None
        self.load_model()

    def is_model_loaded(self) -> bool:
        """Check if model artifact is loaded and ready."""
        return self._bundle is not None and "model" in self._bundle

    def load_model(self) -> bool:
        """Load joblib model bundle if available on disk."""
        if os.path.isfile(self.model_path):
            try:
                self._bundle = joblib.load(self.model_path)
                return True
            except Exception:
                self._bundle = None
                return False
        return False

    def get_model_metrics(self) -> Optional[Dict[str, Any]]:
        """Return evaluation metrics of the trained model."""
        if self.is_model_loaded() and self._bundle:
            return self._bundle.get("metrics")
        return None

    def predict_move(
        self,
        board: chess.Board,
        move: chess.Move,
        eval_before_cp: int = 0,
    ) -> Dict[str, Any]:
        """Predict blunder probability and position risk for a candidate move.

        Args:
            board: chess.Board state before move.
            move: chess.Move to evaluate.
            eval_before_cp: Pre-move engine evaluation in centipawns.

        Returns:
            Dictionary containing prediction probability, risk level, and top features.
        """
        if not self.is_model_loaded() or self._bundle is None:
            return {
                "model_available": False,
                "blunder_probability": 0.0,
                "risk_level": "Unknown (Model Not Loaded)",
                "is_blunder_predicted": False,
                "confidence": 0.0,
                "features": {},
            }

        # 1. Feature Extraction
        feature_dict = extract_features(board, move, eval_before_cp=eval_before_cp)
        feature_names = self._bundle.get("feature_names", FEATURE_COLUMNS)
        feature_vector = np.array([[feature_dict.get(col, 0) for col in feature_names]])

        # 2. Scale features
        scaler = self._bundle.get("scaler")
        if scaler is not None:
            feature_vector_scaled = scaler.transform(feature_vector)
        else:
            feature_vector_scaled = feature_vector

        # 3. Model Prediction
        model = self._bundle["model"]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feature_vector_scaled)[0]
            # Probability of blunder (class 1)
            blunder_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        else:
            pred = model.predict(feature_vector_scaled)[0]
            blunder_prob = 1.0 if pred == 1 else 0.0

        is_blunder = blunder_prob >= 0.50

        # Categorize risk level
        if blunder_prob >= 0.70:
            risk_level = "High Blunder Risk"
            risk_color = "#e74c3c"
        elif blunder_prob >= 0.40:
            risk_level = "Moderate Risk"
            risk_color = "#e67e22"
        elif blunder_prob >= 0.20:
            risk_level = "Low Risk"
            risk_color = "#f1c40f"
        else:
            risk_level = "Safe Move"
            risk_color = "#2ecc71"

        return {
            "model_available": True,
            "blunder_probability": round(blunder_prob * 100, 1),
            "is_blunder_predicted": is_blunder,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "confidence": round(max(blunder_prob, 1.0 - blunder_prob) * 100, 1),
            "features": feature_dict,
        }
