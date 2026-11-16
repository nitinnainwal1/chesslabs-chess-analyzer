# ♟️ ChessLabs — Chess Game Analyzer

**ChessLabs** is an interactive, modular Chess Game Analyzer and coach application built with **Streamlit**, **python-chess**, **Stockfish UCI Engine**, **Scikit-Learn**, and **PyTorch**.

> **Note:** Stockfish is an external engine, not part of this repository. The application detects a locally installed Stockfish binary or standard system package automatically.

---

## 📸 Screenshots

<!-- Screenshots Placeholder -->
> *Add application screenshots here (e.g. Analysis Board, Evaluation Graph, Blunder Risk Attribution).*

---

## 🛠️ Tech Stack

- **UI & Dashboard**: [Streamlit](https://streamlit.io/)
- **Chess Logic & PGN**: [python-chess](https://python-chess.readthedocs.io/)
- **Engine Analysis**: [Stockfish UCI Engine](https://stockfishchess.org/)
- **Visualizations**: [Plotly](https://plotly.com/python/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/), [Joblib](https://joblib.readthedocs.io/)
- **Deep Learning**: [PyTorch](https://pytorch.org/) (2D CNN Spatial Board Evaluator)
- **API Integration**: [Chess.com Public Data API](https://www.chess.com/news/view/published-data-api)
- **Testing**: [Pytest](https://docs.pytest.org/)

---

## 🚀 Installation & Running

### 1. Clone the repository
```bash
git clone https://github.com/nitinnainwal1/chesslabs-chess-analyzer.git
cd chesslabs-chess-analyzer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Run tests
```bash
python -m pytest tests/ -v
```

### 4. Run the Streamlit application
```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```text
├── app.py                      # Main Streamlit Application entrypoint
├── packages.txt                # Linux system packages for Streamlit Cloud (stockfish)
├── requirements.txt            # Pinned Python dependencies
├── chess/                      # Chess domain package (PGN parsing, Stockfish wrapper, Board SVG)
├── ml/                         # Scikit-Learn blunder prediction pipeline & feature extraction
├── dl/                         # PyTorch 2D CNN spatial board evaluation model
├── training/                   # Dataset generation and offline training runners
├── data/                       # Sample PGNs and processed datasets
├── models/                     # Pre-trained ML and DL model artifacts
└── tests/                      # Automated test suite (Pytest)
```
