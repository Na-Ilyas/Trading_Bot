"""
Pipeline Configuration
All hyperparameters in one place for fast experimentation.
"""

# ── Data ──────────────────────────────────────────────
SYMBOL          = "BTC/USDT"
TIMEFRAME       = "1h"
N_CANDLES       = 3000          # ~4 months of hourly data
SYNTHETIC_SEED  = 42            # Reproducibility

# Cross-crypto tickers whose sentiment we also track
CROSS_CRYPTOS   = ["ETH", "SOL", "BNB"]

# ── Feature Engineering ───────────────────────────────
SMA_PERIODS     = [7, 14, 21, 50]
EMA_PERIODS     = [9, 12, 26]
RSI_PERIOD      = 14
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
BB_PERIOD       = 20
BB_STD          = 2
ATR_PERIOD      = 14
LAG_PERIODS     = [1, 2, 3, 6, 12, 24]     # Hourly lags
ROLLING_WINDOWS = [6, 12, 24, 48]           # Rolling stats windows

# ── Splits (temporal, NO shuffle) ─────────────────────
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15

# ── Feature Selection ─────────────────────────────────
USE_BORUTA      = True
USE_LIGHTGBM    = True
BORUTA_MAX_ITER = 50
LGBM_TOP_K      = 30           # Keep top-K features from LightGBM

# ── Sliding Window / Graph ────────────────────────────
WINDOW_SIZE     = 24            # 24 hours lookback

# ── Model ─────────────────────────────────────────────
GCN_UNITS_1     = 32
GCN_UNITS_2     = 16
LSTM_UNITS_1    = 64
LSTM_UNITS_2    = 64
BILSTM_UNITS    = 64
DROPOUT_RATE    = 0.3
LEARNING_RATE   = 0.001
EPOCHS          = 50
BATCH_SIZE      = 64
EARLY_STOP_PAT  = 10

# ── Backtester ────────────────────────────────────────
INITIAL_CAPITAL = 1000.0
TRADE_FEE       = 0.001         # 0.1% Binance fee
LONG_THRESHOLD  = 0.60          # P(up) > this → LONG  (conservative)
SHORT_THRESHOLD = 0.40          # P(up) < this → SHORT (conservative)
POSITION_FRAC   = 0.25          # 25% of capital per trade (adjustable)
