"""
Data Engine
- Generates realistic hourly crypto data (GBM + volatility clustering)
- Computes full suite of technical indicators
- Generates synthetic daily sentiment scores (multi-crypto)
- Assembles the master feature DataFrame
- Supports date-range filtering via start_date/end_date
"""

import numpy as np
import pandas as pd
import config as C


# ═══════════════════════════════════════════════════════
#  1. Synthetic OHLCV Generator  (drop-in for ccxt later)
# ═══════════════════════════════════════════════════════
def _generate_ohlcv(n: int, seed: int, start_price: float = 60000.0,
                    start_date=None) -> pd.DataFrame:
    """Geometric Brownian Motion with GARCH-like vol clustering."""
    rng = np.random.default_rng(seed)
    mu_hourly  = 0.00002           # slight upward drift
    base_sigma = 0.008             # ~0.8 % hourly vol

    prices = np.zeros(n)
    prices[0] = start_price
    sigma = base_sigma

    for t in range(1, n):
        # vol clustering: sigma mean-reverts with shocks
        sigma = 0.95 * sigma + 0.05 * base_sigma + 0.02 * abs(rng.normal())
        ret = mu_hourly + sigma * rng.normal()
        prices[t] = prices[t - 1] * np.exp(ret)

    # Build OHLCV from close prices
    high   = prices * (1 + rng.uniform(0.001, 0.015, n))
    low    = prices * (1 - rng.uniform(0.001, 0.015, n))
    opn    = prices * (1 + rng.uniform(-0.005, 0.005, n))
    volume = rng.lognormal(mean=15, sigma=1.2, size=n)

    if start_date is not None:
        ts = pd.date_range(start=start_date, periods=n, freq="1h")
    else:
        ts = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=n, freq="1h")

    return pd.DataFrame({
        "timestamp": ts,
        "open": opn, "high": high, "low": low,
        "close": prices, "volume": volume
    })


def fetch_ohlcv(start_date=None, end_date=None, n_candles=None) -> pd.DataFrame:
    """Returns hourly OHLCV.  Swap body for ccxt when going live."""
    if start_date is not None and end_date is not None:
        hours = int((end_date - start_date).total_seconds() / 3600)
        hours = max(hours, 100)  # minimum 100 candles
        return _generate_ohlcv(hours, C.SYNTHETIC_SEED, start_date=start_date)
    candles = n_candles if n_candles else C.N_CANDLES
    return _generate_ohlcv(candles, C.SYNTHETIC_SEED)


# ═══════════════════════════════════════════════════════
#  2. Technical Indicators
# ═══════════════════════════════════════════════════════
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]

    # ── Moving Averages ───────────────────────────────
    for p in C.SMA_PERIODS:
        df[f"sma_{p}"] = c.rolling(p).mean()
    for p in C.EMA_PERIODS:
        df[f"ema_{p}"] = c.ewm(span=p, adjust=False).mean()

    # ── RSI ───────────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(C.RSI_PERIOD).mean()
    loss  = (-delta.clip(upper=0)).rolling(C.RSI_PERIOD).mean()
    rs    = gain / (loss + 1e-10)
    df["rsi"] = 100 - 100 / (1 + rs)

    # ── MACD ──────────────────────────────────────────
    ema_fast = c.ewm(span=C.MACD_FAST, adjust=False).mean()
    ema_slow = c.ewm(span=C.MACD_SLOW, adjust=False).mean()
    df["macd"]        = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=C.MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── Bollinger Bands ───────────────────────────────
    mid = c.rolling(C.BB_PERIOD).mean()
    std = c.rolling(C.BB_PERIOD).std()
    df["bb_upper"] = mid + C.BB_STD * std
    df["bb_lower"] = mid - C.BB_STD * std
    df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

    # ── ATR ───────────────────────────────────────────
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - c.shift()).abs(),
        (df["low"]  - c.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(C.ATR_PERIOD).mean()

    # ── OBV (On-Balance Volume) ───────────────────────
    obv = (np.sign(delta) * df["volume"]).fillna(0).cumsum()
    df["obv"] = obv

    # ── Lag Features ──────────────────────────────────
    for lag in C.LAG_PERIODS:
        df[f"ret_lag_{lag}"] = c.pct_change(lag)

    # ── Rolling Statistics ────────────────────────────
    for w in C.ROLLING_WINDOWS:
        rets = c.pct_change()
        df[f"vol_{w}"]      = rets.rolling(w).std()
        df[f"ret_mean_{w}"] = rets.rolling(w).mean()
        df[f"ret_skew_{w}"] = rets.rolling(w).skew()

    # ── Price-relative features ───────────────────────
    df["close_to_sma_50"] = c / df["sma_50"] - 1  if "sma_50" in df else 0
    df["high_low_range"]  = (df["high"] - df["low"]) / c

    return df


# ═══════════════════════════════════════════════════════
#  3. Synthetic Multi-Crypto Sentiment
# ═══════════════════════════════════════════════════════
def generate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates daily sentiment scores for BTC + cross-cryptos.
    In production: replace with FinBERT on RSS/Twitter headlines.
    Sentiment = mean of (positive_prob - negative_prob) across headlines.
    """
    df = df.copy()
    rng = np.random.default_rng(C.SYNTHETIC_SEED + 1)

    # Daily granularity → forward-fill to hourly
    n_days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days + 1
    dates  = pd.date_range(start=df["timestamp"].iloc[0].normalize(),
                           periods=n_days, freq="D")

    # BTC sentiment: slightly correlated with previous day's return
    daily_close = df.set_index("timestamp")["close"].resample("D").last().dropna()
    daily_ret   = daily_close.pct_change().fillna(0).values

    btc_sent = np.zeros(len(dates))
    for i in range(len(dates)):
        # Use PREVIOUS day's return to avoid intra-day look-ahead
        if i < len(daily_ret):
            prev_ret = daily_ret[i - 1] if i >= 1 else 0.0
            btc_sent[i] = np.clip(0.5 + 0.3 * prev_ret + 0.15 * rng.normal(), 0, 1)
        else:
            btc_sent[i] = np.clip(0.5 + 0.15 * rng.normal(), 0, 1)

    sent_df = pd.DataFrame({"date": dates, "sentiment_btc": btc_sent})

    # Cross-crypto sentiments (partially correlated with BTC)
    for sym in C.CROSS_CRYPTOS:
        noise = 0.2 * rng.normal(size=len(dates))
        sent_df[f"sentiment_{sym.lower()}"] = np.clip(btc_sent + noise, 0, 1)

    # Merge to hourly: forward-fill daily sentiment
    df["date"] = df["timestamp"].dt.normalize()
    df = df.merge(sent_df, on="date", how="left")
    for col in sent_df.columns:
        if col.startswith("sentiment"):
            df[col] = df[col].ffill().bfill()
    df.drop(columns=["date"], inplace=True)

    return df


# ═══════════════════════════════════════════════════════
#  4. Master Feature Assembly
# ═══════════════════════════════════════════════════════
def build_features(start_date=None, end_date=None, n_candles=None) -> pd.DataFrame:
    """Full pipeline: fetch → indicators → sentiment → clean."""
    df = fetch_ohlcv(start_date=start_date, end_date=end_date, n_candles=n_candles)
    df = add_technical_indicators(df)
    df = generate_sentiment(df)

    # Target: binary direction (1 = price goes up next hour)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Drop rows with NaN from rolling calculations and last row (no target)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


if __name__ == "__main__":
    df = build_features()
    print(f"Shape: {df.shape}")
    print(f"Features: {[c for c in df.columns if c not in ['timestamp','target']]}")
    print(f"Target balance: {df['target'].value_counts().to_dict()}")
    print(df.tail(3))
