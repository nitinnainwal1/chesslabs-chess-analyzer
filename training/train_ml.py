"""ChessLabs - ML Model Training Runner."""

import os
import sys

# Ensure workspace root in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.train import train_and_evaluate_models


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(base_dir, "data", "processed", "chess_ml_dataset.csv")
    out_model = os.path.join(base_dir, "models", "move_quality_model.joblib")

    print("========================================")
    print("ChessLabs: Training Machine Learning Model")
    print("========================================")
    train_and_evaluate_models(csv_file, out_model)


if __name__ == "__main__":
    main()
