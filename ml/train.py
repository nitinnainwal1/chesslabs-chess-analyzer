import os
import sys

# Ensure workspace root in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from ml.preprocessing import split_and_scale_data


def train_and_evaluate_models(
    csv_path: str,
    model_save_path: str,
) -> Dict[str, Any]:
    """Load dataset, train Logistic Regression and Random Forest, compare metrics, and save best model.

    Args:
        csv_path: Path to input CSV dataset.
        model_save_path: Destination path for saving .joblib model bundle.

    Returns:
        Dictionary containing trained model results and evaluation metrics.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale_data(
        df, target_column="is_blunder", test_size=0.25, random_state=42
    )

    print(f"Dataset split: {len(X_train)} training positions, {len(X_test)} testing positions.")
    print(f"Target distribution (Train): {np.bincount(y_train)}, (Test): {np.bincount(y_test)}")

    # 1. Baseline Model: Logistic Regression
    lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    lr_metrics = {
        "model_name": "Logistic Regression (Baseline)",
        "accuracy": round(float(accuracy_score(y_test, lr_preds)), 4),
        "precision": round(float(precision_score(y_test, lr_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, lr_preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, lr_preds, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, lr_preds).tolist(),
    }

    # 2. Advanced Model: Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    rf_metrics = {
        "model_name": "Random Forest Classifier",
        "accuracy": round(float(accuracy_score(y_test, rf_preds)), 4),
        "precision": round(float(precision_score(y_test, rf_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, rf_preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, rf_preds, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, rf_preds).tolist(),
    }

    print("\n--- Model Evaluation Results ---")
    print(f"Logistic Regression -> Accuracy: {lr_metrics['accuracy']:.2%}, F1: {lr_metrics['f1']:.4f}")
    print(f"Random Forest       -> Accuracy: {rf_metrics['accuracy']:.2%}, F1: {rf_metrics['f1']:.4f}")

    # Select best model based on F1-score
    if rf_metrics["f1"] >= lr_metrics["f1"]:
        best_model = rf_model
        best_metrics = rf_metrics
        feature_importances = dict(zip(feature_names, [round(float(v), 4) for v in rf_model.feature_importances_]))
    else:
        best_model = lr_model
        best_metrics = lr_metrics
        feature_importances = dict(zip(feature_names, [round(float(v), 4) for v in lr_model.coef_[0]]))

    best_metrics["feature_importances"] = feature_importances

    # Save model artifact bundle
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    bundle = {
        "model": best_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "metrics": best_metrics,
        "comparison": [lr_metrics, rf_metrics],
    }
    joblib.dump(bundle, model_save_path)
    print(f"[SUCCESS] Best model ({best_metrics['model_name']}) saved to: {model_save_path}")

    return bundle


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(base_dir, "data", "processed", "chess_ml_dataset.csv")
    out_model = os.path.join(base_dir, "models", "move_quality_model.joblib")
    train_and_evaluate_models(csv_file, out_model)
