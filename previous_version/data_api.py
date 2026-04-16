
import os
import pandas as pd
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Shared data cache with new_version pipeline
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'new_version', 'data')
CACHE_FILE = os.path.join(DATA_DIR, 'btcusdt_1h.csv')


def fetch_historical_data(symbol='BTC/USDT', timeframe='1h', total_candles=5000,
                          start_date=None, end_date=None):
    """Load BTC/USDT data from local CSV cache (shared with new_version pipeline).
    Falls back to Binance API if cache doesn't exist."""

    if os.path.exists(CACHE_FILE):
        df = pd.read_csv(CACHE_FILE, parse_dates=['timestamp'])
        logging.info(f"Loaded {len(df)} candles from cache")

        if start_date is not None and end_date is not None:
            mask = ((df['timestamp'] >= pd.Timestamp(start_date)) &
                    (df['timestamp'] <= pd.Timestamp(end_date)))
            df = df[mask].reset_index(drop=True)
        else:
            df = df.tail(total_candles).reset_index(drop=True)

        if len(df) > 0:
            logging.info(f"Using {len(df)} candles: "
                         f"{df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
            logging.info(f"Price range: ${df['close'].iloc[0]:,.2f} -> "
                         f"${df['close'].iloc[-1]:,.2f}")
            return df
        logging.warning("No candles in requested range")

    # Fallback to API if no cache
    logging.info("No local cache found, fetching from Binance API...")
    return _fetch_from_api(symbol, timeframe, total_candles)


def _fetch_from_api(symbol='BTC/USDT', timeframe='1h', total_candles=5000):
    """Original API fetch logic as fallback."""
    try:
        import ccxt
    except ImportError:
        logging.error("ccxt not installed. Run: pip install ccxt")
        return pd.DataFrame()

    exchange = ccxt.binance({'enableRateLimit': True})
    limit = 1000
    all_ohlcv = []

    try:
        latest = exchange.fetch_ohlcv(symbol, timeframe, limit=2)
        end_time = latest[-1][0]
    except Exception as e:
        logging.error(f"Error fetching initial data: {e}")
        return pd.DataFrame()

    logging.info(f"Fetching {total_candles} candles for {symbol}...")

    for _ in range(total_candles // limit + 1):
        if len(all_ohlcv) >= total_candles:
            break
        try:
            since = end_time - (limit * 60 * 60 * 1000)
            data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)
            if not data:
                break
            all_ohlcv = data + all_ohlcv
            end_time = data[0][0] - 1
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            logging.warning(f"API error: {e}. Retrying in 5s...")
            time.sleep(5)

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.drop_duplicates(subset='timestamp', inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df.tail(total_candles)
