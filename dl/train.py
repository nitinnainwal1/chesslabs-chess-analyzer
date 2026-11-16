"""ChessLabs - Deep Learning PyTorch Training Pipeline.

Trains the ChessCNN on real chess board positions extracted from the dataset.
Saves model weights and evaluation metrics to disk.
"""

import os
import time
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dl.dataset import fen_to_tensor, ChessPositionDataset
from dl.model import ChessCNN


DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "processed",
    "chess_ml_dataset.csv",
)

DEFAULT_MODEL_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "chess_position_model.pth",
)


def load_dl_training_data(csv_path: str = DEFAULT_DATASET_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """Load positions and target centipawn evaluations from CSV dataset.

    Returns:
        (tensors array of shape (N, 12, 8, 8), targets array of shape (N,))
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Run dataset generator first.")

    df = pd.read_csv(csv_path)
    
    # Filter rows with valid FEN and evaluation
    if "fen" not in df.columns or "eval_after" not in df.columns:
        raise ValueError("Dataset must contain 'fen' and 'eval_after' columns.")

    tensors = []
    targets = []

    for _, row in df.iterrows():
        fen = str(row["fen"])
        eval_cp = float(row["eval_after"])
        
        # Normalize eval into [-1.0, 1.0] by dividing by 1000.0 and clipping
        norm_target = np.clip(eval_cp / 1000.0, -1.0, 1.0)
        
        try:
            tensor = fen_to_tensor(fen)
            tensors.append(tensor)
            targets.append(norm_target)
        except Exception:
            continue

    return np.array(tensors, dtype=np.float32), np.array(targets, dtype=np.float32)


def train_dl_model(
    csv_path: str = DEFAULT_DATASET_PATH,
    model_save_path: str = DEFAULT_MODEL_SAVE_PATH,
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> Dict[str, Any]:
    """Train the PyTorch ChessCNN model on board positions and save checkpoint.

    Returns:
        Dictionary with training metrics, sample count, and validation loss.
    """
    print(f"Loading training data from {csv_path}...")
    tensors, targets = load_dl_training_data(csv_path)
    dataset = ChessPositionDataset(tensors, targets)

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ChessCNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    print(f"Training ChessCNN on {train_size} training samples ({val_size} validation)...")
    start_time = time.time()
    
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_loss / train_size
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_outputs = model(val_x)
                loss = criterion(val_outputs, val_y)
                val_loss += loss.item() * val_x.size(0)

        epoch_val_loss = val_loss / val_size
        val_losses.append(epoch_val_loss)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch}/{epochs}] - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    elapsed_sec = time.time() - start_time

    # Save artifact
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metrics": {
            "train_samples": train_size,
            "val_samples": val_size,
            "final_train_loss": round(train_losses[-1], 4),
            "final_val_loss": round(val_losses[-1], 4),
            "epochs": epochs,
            "elapsed_seconds": round(elapsed_sec, 2),
        },
    }
    torch.save(checkpoint, model_save_path)
    print(f"Model successfully saved to {model_save_path} in {elapsed_sec:.2f}s!")

    return checkpoint["metrics"]


if __name__ == "__main__":
    metrics = train_dl_model()
    print("Training finished with metrics:", metrics)
