import ccxt
import pandas as pd

def fetch_current_market_data(symbol='BTC/USDT', timeframe='1h'):
    exchange = ccxt.binance()
    # Fetching the most recent 100 candles for current prediction
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df