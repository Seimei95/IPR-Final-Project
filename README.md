# Noise Resilience of Dimensionality Reduction for Human Activity Recognition
CSE 555, Fall 2026, Track 2. Team: Jeevan Ankith Pally, Deepesh Chhetri, Sri Lakshmi Vaddi.

## Setup
1. Install uv: https://docs.astral.sh/uv/
2. Download the official dataset zip from
   https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
   Unzip it, then unzip the inner `UCI HAR Dataset.zip`, and put the folder at
   `data/UCI HAR Dataset/` (so `data/UCI HAR Dataset/train/X_train.txt` exists).
   Do not use the `ucimlrepo` package: it drops the subject-wise train/test split.
3. `uv sync`

## Run
- Tests: `uv run pytest -q`  (28 tests)
- Everything in the report: `uv run python run_experiments.py`  (about 5 minutes)
- Quick smoke run: `uv run python run_experiments.py --quick`
- Notebook: `uv run jupyter lab` and open `report.ipynb` (it only calls the code above)

Outputs: `results/` (CSV/JSON tables) and `figures/` (PNG).
All randomness is seeded, so two runs on the same machine give identical files.

## Layout
- `har_noise/data.py`: loads the official split, de-duplicates the 42 repeated feature names, builds the 17 signal groups
- `har_noise/corruption.py`: Gaussian noise, signal-group dropout, OOD injection (test data only)
- `har_noise/projection.py`: PCA and LDA from scratch (NumPy/SciPy), PCA-then-LDA, FastICA wrapper
- `har_noise/classifiers.py`: k-NN, linear SVM, Gaussian NB; subject-grouped 5-fold CV for hyperparameters
- `har_noise/metrics.py`: Fisher's criterion, macro P/R/F1, degradation slope, breakdown point
- `har_noise/experiment.py`: the corruption x level x seed x projection x classifier sweep
- `har_noise/plots.py`: all figures
- `run_experiments.py`: single entry point
- `tests/test_har_noise.py`: unit tests, incl. PCA and LDA checked against scikit-learn

Dataset: Reyes-Ortiz et al., UCI ML Repository, doi:10.24432/C54S4K, CC BY 4.0.