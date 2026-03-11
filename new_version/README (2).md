# Crypto Trading Pipeline: GCN-BiLSTM + Feature Selection

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION                          │
│  Binance OHLCV (1H) ─┬─ BTC/USDT                           │
│  RSS/Twitter News ────┤  ETH, SOL, BNB sentiment            │
│  On-chain metrics ────┘  (multi-crypto cross-signals)       │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                          │
│  SMA(7,14,21,50) │ EMA(9,12,26) │ RSI(14) │ MACD           │
│  Bollinger Bands  │ ATR(14)      │ OBV     │ Lag returns    │
│  Rolling vol/skew │ Cross-crypto sentiment scores           │
│  45+ raw features                                            │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          TEMPORAL SPLIT (70/15/15, NO shuffle)               │
│  Scaler FIT on train only → transform val/test              │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│         FEATURE SELECTION (on TRAINING data only)            │
│  Boruta (wrapper, RF-based) ──┐                              │
│  LightGBM (embedded, top-K) ──┴─ UNION → 30 features        │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              SLIDING WINDOW + GRAPH                           │
│  Window = 24 hours    Temporal chain adjacency matrix        │
│  A_norm = D^{-½} (A+I) D^{-½}  (symmetric normalization)   │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              HYBRID GCN-BiLSTM MODEL                         │
│                                                              │
│  ┌───────────────┐        ┌──────────────────────┐          │
│  │  GCN Branch   │        │   BiLSTM Branch      │          │
│  │  GCN(32)→Drop │        │   LSTM(64)→BN        │          │
│  │  GCN(16) seq. │        │   LSTM(64)→BN        │          │
│  │  GlobalAvgPool│        │   BiLSTM(64)→Drop    │          │
│  └───────┬───────┘        └──────────┬───────────┘          │
│          └──────────┬────────────────┘                       │
│                  Concat                                      │
│              Dense(64) → Dense(32) → Sigmoid                │
│              Output: P(price_up)                             │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKTESTER                                 │
│  P > 0.60 → LONG    P < 0.40 → SHORT    else → HOLD        │
│  Position size: 25% of capital (adjustable)                  │
│  Metrics: Directional Acc, PnL, Sharpe, Max Drawdown        │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters in one place |
| `data_engine.py` | OHLCV fetch + technical indicators + sentiment |
| `feature_selector.py` | Boruta + LightGBM (train-only) |
| `hybrid_model.py` | GCN-BiLSTM with custom GCN layer |
| `backtester.py` | Trading sim with adjustable position sizing |
| `run_pipeline.py` | Main orchestrator (Steps 1-8) |

## Critical Fixes vs Original Code

| Issue | Old Code | Fixed Pipeline |
|-------|----------|----------------|
| Data leakage | `scaler.fit_transform(ALL_DATA)` | `scaler.fit(train)` only |
| GCN bug | Parallel GCN layers | Sequential (layer2 reads layer1 output) |
| Target | Absolute price (non-stationary) | Direction classification (up/down) |
| Sentiment | Single scalar for all rows | Daily per-crypto scores, forward-filled |
| Feature selection | None | Boruta + LightGBM on train only |
| Position sizing | 100% capital per trade | 25% adjustable fraction |

## Going Live

Replace the synthetic data generator in `data_engine.py`:

```python
# In fetch_ohlcv(), swap the body:
import ccxt
def fetch_ohlcv():
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(C.SYMBOL, C.TIMEFRAME, limit=C.N_CANDLES)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# In generate_sentiment(), swap with FinBERT:
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import feedparser, torch

def get_live_sentiment(headlines):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    return (probs[:, 2] - probs[:, 0]).mean().item()  # positive - negative
```

## Quick Start

```bash
pip install tensorflow scikit-learn pandas numpy boruta lightgbm
cd crypto_pipeline
python run_pipeline.py
```

Adjust `config.py` to tune thresholds, position sizing, and feature selection.
