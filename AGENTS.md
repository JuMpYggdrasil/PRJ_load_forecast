# AGENTS.md — PRJ_load_forecast

## Project

LSTM time-series load forecasting for EGAT (Electricity Generating Authority of Thailand). Predicts electrical load (kW) at 15-minute granularity from CSV data.

## Entrypoint

- `Time_Series_Prediction.py` — main script, function `time_series_prediction()`
- `loop_prediction.py` — hyperparameter sweeper importing the function above

## Run commands

```powershell
.venv\Scripts\python Time_Series_Prediction.py
.venv\Scripts\python loop_prediction.py
.venv\Scripts\python Time_Series_Prediction_profiling.py
.venv\Scripts\python check_cuda.py          # verify CUDA/GPU setup
```

## Critical quirks

- **Date format varies by CSV**: `ratch_data.csv` uses `%d/%m/%Y %H:%M`; `combined_data.csv` and `number_data.csv` use `%H.%M`. The active line in `Time_Series_Prediction.py` defaults to the `ratch_data` format. Swap commented lines if switching datasets.
- **Model cache**: `trained_lstm_model.pth` is loaded automatically if present. Delete it to force retraining. `loop_prediction.py` does this per iteration.
- **Savgol filter**: `window_length` must be odd, `polyorder` < `window_length`.
- **Output plots** written to repo root: `time_series_prediction.png`, `filter_data.png`, `error_distribution.png`, `sliding_window_sample_*.png`
- **Results logged** to `model_summary_results.csv` (MSE, MAE, R², MAPE, Accuracy, hyperparameters)

## Dependencies

Python venv at `.venv\`. Install with:
```
.venv\Scripts\pip install -r requirements.txt
```

Key: PyTorch 2.5.1+cu121, pandas 2.3.1, scikit-learn 1.7.1, matplotlib 3.10.3.

## Project context

`.opencode/rules.md` contains Thai-language business-agent rules for this EGAT energy innovation project. Check it for context-specific conventions.
