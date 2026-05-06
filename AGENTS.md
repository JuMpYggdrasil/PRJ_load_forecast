# AGENTS.md

## Project
LSTM time-series load forecasting (Python 3.10, PyTorch CUDA 12.1). Flat directory, no packages, no tests, no CI.

## Setup
- venv at `.venv` (Python 3.10.11)
- `pip install -r requirements.txt`
- GPU optional; code auto-falls back to CPU
- `python check_cuda.py` — diagnostics for GPU, PyTorch, and (if expanding into Deep RL) gymnasium/stable_baselines3

## Commands
- **Train + evaluate:** `python Time_Series_Prediction.py`
  - Reads `ratch_data.csv` (`Date`, `Load`; `dd/mm/yyyy HH:MM`; 15-min intervals)
  - Savitzky-Golay smoothing (window=11, polyorder=3), 67/33 split, batch 1024, 1000 epochs, lr 0.01
  - Loads `trained_lstm_model.pth` if it exists, otherwise trains and saves it
  - Outputs: `filter_data.png`, `time_series_prediction.png`, `sliding_window_sample_*.png`, appends to `model_summary_results.csv`
- **Hyperparameter sweep:** `python loop_prediction.py` — iterates all combos of seq_length, hidden_size, stacked_size, dropout; deletes `trained_lstm_model.pth` between runs (intentional)
- **Profiling:** `python Time_Series_Prediction_profiling.py` — uses `combined_data.csv`, writes TensorBoard traces to `log/profiler/`
- **Old/unused:** `backup.py`, `timeserie_predict_old.py` — legacy versions, ignore unless explicitly needed

## Key gotchas
- No `.gitignore` — `.venv/`, `__pycache__/`, `.pth`, `.png`, `.csv`, `log/profiler/*.json` are all git-tracked. Do not commit venv or large artifacts unless asked.
- Default hyperparams in `Time_Series_Prediction.py` `__main__`: `seq_length=128, hidden_size=16, stacked_size=2, dropout=0`
- `Time_Series_Prediction_profiling.py` uses `combined_data.csv` and a different date format (`%d/%m/%Y %H.%M` with dot) — do not confuse with `ratch_data.csv`
- Profiler traces viewed via `tensorboard --logdir log/profiler`

## No test/lint/typecheck
No tests, linter, formatter, or type checker. Verify by running scripts and checking console output/plots.
