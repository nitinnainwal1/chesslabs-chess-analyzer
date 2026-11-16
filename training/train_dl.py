"""ChessLabs - Deep Learning Training Runner.

Convenience CLI script to train the PyTorch Deep Learning board position evaluation model.
"""

import sys
import os

# Ensure root is on python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dl.train import train_dl_model


def main() -> None:
    print("=" * 60)
    print("ChessLabs: Training Deep Learning Board Evaluation Model (PyTorch)")
    print("=" * 60)
    
    metrics = train_dl_model(epochs=20, batch_size=32, learning_rate=0.001)
    
    print("\n" + "=" * 60)
    print(f"Deep Learning Model Training Complete!")
    print(f"Artifact: models/chess_position_model.pth")
    print(f"Final Validation Loss: {metrics['final_val_loss']}")
    print(f"Training Time: {metrics['elapsed_seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
