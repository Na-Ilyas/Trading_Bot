# Bitcoin Intraday Trading Bot — GCN-BiLSTM Hybrid Model

An end-to-end deep learning pipeline for hourly cryptocurrency direction prediction and automated backtesting. The system combines a **Graph Convolutional Network (GCN)** with a **Bidirectional LSTM** to capture both graph-structured temporal dependencies and sequential patterns in price data.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Steps](#pipeline-steps)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Backtesting & Trading Logic](#backtesting--trading-logic)
- [Trade Log Columns](#trade-log-columns)
- [Output Artifacts](#output-artifacts)
- [Comparative Mode](#comparative-mode)
- [Configuration](#configuration)
- [Model Improvement Flags](#model-improvement-flags)

---

## Architecture Overview

```
Raw OHLCV Data (hourly)
    |
    v
Technical Indicators (45 features: SMA, EMA, RSI, MACD, Bollinger, ATR, OBV, lags, rolling stats)
    |
    v
Multi-Crypto Sentiment (synthetic daily scores for BTC, ETH, SOL, BNB)
    |
    v
Temporal Train/Val/Test Split (70/15/15, NO shuffle)
    |
    v
Feature Selection (Boruta + LightGBM on training data ONLY)
    |
    v
StandardScaler (fit on train, transform val/test)
    |
    v
Sliding Windows (24-hour lookback)
    |
    v
+-------------------+    +-------------------+
| GCN Branch        |    | BiLSTM Branch     |
| GCN(32) -> GCN(16)|    | LSTM(64) x2       |
| GlobalAvgPool     |    | BiLSTM(64)        |
+--------+----------+    +--------+----------+
         |                         |
         +-------Concatenate-------+
                    |
                    v
             Dense(64) -> Dense(32) -> Sigmoid
                    |
                    v
            P(price_up) in [0, 1]
                    |
                    v
    Adaptive Threshold Calibration (from validation set)
                    |
                    v
         LONG / SHORT / HOLD decision
                    |
                    v
          Backtest Simulation with PnL tracking
```

## Project Structure

```
new_version/
  config.py             # All hyperparameters and feature flags
  data_engine.py        # OHLCV generation, technical indicators, sentiment
  feature_selector.py   # Boruta + LightGBM feature selection
  hybrid_model.py       # GCN + BiLSTM model (with optional improvements)
  backtester.py         # Trading simulator with enriched trade logging
  baseline_models.py    # LSTM, RNN, XGBoost, ARIMA, BuyAndHold baselines
  run_pipeline.py       # Main orchestrator (argparse, charts, comparative mode)
  README.md             # This file
```

## Installation

**Python 3.10+** required. Install dependencies:

```bash
pip install numpy pandas scikit-learn tensorflow lightgbm boruta xgboost statsmodels matplotlib
```

## Usage

### Default mode (synthetic data, last 3000 hours)

```bash
python run_pipeline.py
```

### Custom date range

```bash
python run_pipeline.py --start 06.10.2025 --end 06.02.2026
```

Date format: `DD.MM.YYYY`. The pipeline generates synthetic data covering the specified period.

### Comparative mode (all models)

```bash
python run_pipeline.py --compare
```

Trains and backtests 6 models on the same data: Hybrid GCN-BiLSTM, standalone LSTM, SimpleRNN, XGBoost, ARIMA, and BuyAndHold.

### Combined

```bash
python run_pipeline.py --start 06.10.2025 --end 06.02.2026 --compare
```

## Pipeline Steps

| Step | Description | File |
|------|-------------|------|
| 1 | Build feature matrix (OHLCV + indicators + sentiment) | `data_engine.py` |
| 2 | Temporal train/val/test split (70/15/15, no shuffle) | `run_pipeline.py` |
| 3 | Feature selection on training data only (Boruta + LightGBM) | `feature_selector.py` |
| 4 | Scale features (fit on train, transform val/test) | `run_pipeline.py` |
| 5 | Create sliding windows (24h) + graph adjacency matrix | `run_pipeline.py` |
| 6 | Train hybrid GCN-BiLSTM model | `hybrid_model.py` |
| 7 | Predict on test set + calibrate thresholds from validation | `run_pipeline.py` |
| 8 | Backtest simulation with position sizing | `backtester.py` |
| 9 | Save artifacts, charts, and run summary | `run_pipeline.py` |

## Feature Engineering

**45 raw features** computed from hourly OHLCV:

| Category | Features | Count |
|----------|----------|-------|
| Moving Averages | SMA(7,14,21,50), EMA(9,12,26) | 7 |
| Momentum | RSI(14), MACD, MACD Signal, MACD Histogram | 4 |
| Volatility | Bollinger Bands (upper, lower, %B), ATR(14) | 4 |
| Volume | On-Balance Volume (OBV) | 1 |
| Lag Returns | pct_change(1,2,3,6,12,24) | 6 |
| Rolling Stats | volatility, mean return, skewness over (6,12,24,48)h windows | 12 |
| Price Relative | close/SMA50 ratio, high-low range | 2 |
| Sentiment | Daily synthetic scores for BTC, ETH, SOL, BNB | 4 |
| Raw OHLCV | open, high, low, close, volume | 5 |

After feature selection (Boruta + LightGBM union), typically **30-33 features** are retained.

**Target variable:** Binary direction — `1` if next hour's close > current close, `0` otherwise.

## Model Architecture

### GCN Branch
- Temporal adjacency matrix: each hour connected to its neighbors (chain graph)
- Normalized: D^{-1/2}(A+I)D^{-1/2}
- GCN Layer 1: 32 units, ReLU
- GCN Layer 2: 16 units, ReLU
- Global Average Pooling -> 16-dim vector

### BiLSTM Branch
- LSTM(64) with BatchNorm x2 (stacked)
- Bidirectional LSTM(64) -> 128-dim vector

### Fusion
- Concatenate [16-dim GCN, 128-dim BiLSTM] -> 144-dim
- Dense(64, ReLU) -> Dropout -> Dense(32, ReLU) -> Dense(1, Sigmoid)

### Training
- Optimizer: Adam (lr=0.001)
- Loss: Binary Crossentropy
- Early Stopping: patience=10, restore best weights
- ReduceLROnPlateau: factor=0.5, patience=5

### Adaptive Threshold Calibration
Instead of fixed LONG/SHORT thresholds (0.6/0.4), the pipeline calibrates thresholds from validation set predictions using percentiles. This ensures the model trades at a consistent frequency regardless of how wide or narrow the probability distribution is.

## Backtesting & Trading Logic

**Decision rule:**
- `P(up) > calibrated_long_threshold` -> **LONG** (buy, profit if price rises)
- `P(up) < calibrated_short_threshold` -> **SHORT** (sell, profit if price falls)
- Otherwise -> **HOLD** (no position)

**Position sizing:** Fixed fraction (25% of capital per trade by default). Optional Kelly criterion available via `USE_KELLY=True`.

**Fees:** 0.1% per trade (Binance spot fee).

**Metrics computed:**
- Total Return %, Final Capital, Max Equity
- Sharpe Ratio (annualized from hourly returns)
- Max Drawdown %
- Win Rate, Profit Factor, Avg Trade PnL
- Biggest Win/Loss, Max Win/Loss Streaks
- Total Fees Paid

## Trade Log Columns

Each run produces a `trade_log.csv` with 23 columns:

| Column | Description |
|--------|-------------|
| `symbol` | Cryptocurrency pair (e.g., BTC/USDT) |
| `timestamp` | Hourly timestamp of the signal |
| `trade_number` | Sequential trade counter (0 for HOLD) |
| `action` | LONG, SHORT, or HOLD |
| `entry_price` | Close price at signal hour |
| `exit_price` | Close price at next hour |
| `price_change_pct` | Raw price change between entry and exit (%) |
| `prob` | Model's P(price goes up), range [0, 1] |
| `pred_direction` | Binary prediction (1=up, 0=down) |
| `actual_direction` | Ground truth direction |
| `correct` | Whether prediction matched reality |
| `position_size_pct` | % of capital allocated to this trade |
| `trade_capital` | Dollar amount committed to this trade |
| `pnl_pct` | Trade profit/loss as percentage (after fees) |
| `pnl_amount` | Trade profit/loss in dollars |
| `fee_paid` | Transaction fee paid in dollars |
| `capital_after` | Portfolio value after this step |
| `peak_capital` | All-time high portfolio value so far |
| `drawdown_pct` | Current drawdown from peak (%) |
| `cumulative_pnl` | Running total PnL in dollars |
| `cumulative_trades` | Running count of executed trades |
| `cumulative_win_rate_pct` | Running win rate across all trades so far |
| `portfolio_return_pct` | Total portfolio return from initial capital (%) |

## Output Artifacts

Each run creates a timestamped directory `output/run_YYYYMMDD_HHMMSS/` with:

### Data Files
| File | Description |
|------|-------------|
| `run_summary.json` | Full run metadata, model stats, backtest report, config snapshot |
| `trade_log.csv` | Per-step trade log with 23 columns |
| `equity_curve.csv` | Capital value at each step |
| `selected_features.csv` | Features chosen by Boruta + LightGBM |
| `training_history.csv` | Per-epoch loss, accuracy, val_loss, val_accuracy, learning_rate |

### Charts
| Chart | Description |
|-------|-------------|
| `equity_curve.png` | Portfolio equity line with drawdown shading |
| `pnl_distribution.png` | Histograms of per-trade PnL in both % and $ |
| `training_curves.png` | Loss and accuracy curves (train vs validation) |
| `cumulative_pnl.png` | Running total profit/loss over time with green/red shading |
| `prediction_confidence.png` | Scatter plot of model probabilities colored by action taken |
| `rolling_win_rate.png` | 20-trade rolling win rate vs 50% baseline |
| `action_breakdown.png` | Pie charts: action distribution + win/loss ratio |

### Comparative Mode Output
| File | Description |
|------|-------------|
| `comparative_results.csv` | Side-by-side metrics for all 6 models |
| `model_comparison.png` | Grouped bar chart comparing key metrics |

## Comparative Mode

When run with `--compare`, the pipeline trains 6 models on identical data:

| Model | Type | Input Format |
|-------|------|-------------|
| Hybrid (GCN-BiLSTM) | Deep Learning | 3D windowed + adjacency matrix |
| LSTM | Deep Learning | 3D windowed |
| RNN | Deep Learning | 3D windowed |
| XGBoost | Gradient Boosting | 2D flattened windows |
| ARIMA | Time Series | Raw close prices |
| BuyAndHold | Baseline | Always LONG |

Each model gets its own calibrated thresholds and independent backtest. Results are saved to `comparative_results.csv` with columns: Model, Test Accuracy, Trade Accuracy, Return %, Sharpe, Max Drawdown, Win Rate, Trade Count, and Training Time.

## Configuration

All hyperparameters are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_CANDLES` | 3000 | Number of hourly candles (~4 months) |
| `WINDOW_SIZE` | 24 | Lookback window in hours |
| `TRAIN/VAL/TEST_RATIO` | 70/15/15 | Temporal split ratios |
| `GCN_UNITS_1/2` | 32/16 | GCN layer dimensions |
| `LSTM_UNITS_1/2` | 64/64 | Stacked LSTM dimensions |
| `BILSTM_UNITS` | 64 | Bidirectional LSTM dimension |
| `DROPOUT_RATE` | 0.3 | Dropout probability |
| `LEARNING_RATE` | 0.001 | Adam optimizer learning rate |
| `EPOCHS` | 50 | Max training epochs |
| `BATCH_SIZE` | 64 | Mini-batch size |
| `EARLY_STOP_PAT` | 10 | Early stopping patience |
| `INITIAL_CAPITAL` | 1000 | Starting portfolio value ($) |
| `TRADE_FEE` | 0.001 | Per-trade fee (0.1%) |
| `LONG_THRESHOLD` | 0.60 | Config threshold (overridden by calibration) |
| `SHORT_THRESHOLD` | 0.40 | Config threshold (overridden by calibration) |
| `POSITION_FRAC` | 0.25 | Fraction of capital per trade |

## Model Improvement Flags

Optional architectural improvements, all **disabled by default**. Enable in `config.py` for experimentation:

| Flag | Default | Description |
|------|---------|-------------|
| `USE_ATTENTION` | False | Temporal attention mechanism on BiLSTM outputs |
| `USE_RESIDUAL` | False | Residual (skip) connection in fusion layer |
| `LABEL_SMOOTHING` | 0.0 | Binary crossentropy label smoothing (set to 0.1 to enable) |
| `USE_LEARNABLE_ADJ` | False | Learnable adjacency matrix (experimental) |
| `USE_KELLY` | False | Kelly criterion dynamic position sizing |

These features add model capacity but require careful tuning. On small datasets (~2000 training windows), the base architecture performs better. Enable when working with larger or real-world datasets.

---

## Data Leakage Prevention

The pipeline is designed to prevent data leakage at every stage:

1. **Temporal split** with no shuffling — future data never contaminates training
2. **Feature selection** runs on training data only
3. **Scaler** is fit on training data, then applied to val/test
4. **Sentiment features** use only the previous day's return (not same-day)
5. **Technical indicators** are all backward-looking (SMA, EMA, RSI, etc.)
6. **Sliding windows** are created after the split, within each partition

## Going Live

To switch from synthetic to real data:

1. Replace `fetch_ohlcv()` in `data_engine.py` with a CCXT exchange call
2. Replace `generate_sentiment()` with FinBERT on real news/social media
3. Add live order execution via exchange API
4. The rest of the pipeline (indicators, feature selection, model, backtest) works unchanged
