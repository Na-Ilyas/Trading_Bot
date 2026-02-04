import ccxt
import feedparser
import pandas as pd

def get_binance_ohlcv(symbol='BTC/USDT', timeframe='1h', limit=500):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def get_crypto_news():
    # RSS Feeds identified in major literature
    feeds = ['https://www.coindesk.com/arc/outboundfeeds/rss/']
    news_items = []
    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            news_items.append({'title': entry.title, 'date': entry.published})
    return pd.DataFrame(news_items)


def fetch_pipeline_data(symbol='BTC/USDT'):
    # 1. Market Data from Binance
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
    
    # 2. News Data from CoinDesk/CoinTelegraph RSS
    news_feed = feedparser.parse('https://www.coindesk.com/arc/outboundfeeds/rss/')
    latest_headlines = [entry.title for entry in news_feed.entries[:10]]
    
    return ohlcv, latest_headlines