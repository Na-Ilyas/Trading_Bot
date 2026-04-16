import pandas as pd
import pandas_ta as ta
import numpy as np

def feature_engineering(df):
    """Generates technical indicators and log returns WITHOUT data leakage."""
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    df['RSI_14'] = ta.rsi(df['close'], length=14)
    df['SMA_20'] = ta.sma(df['close'], length=20)
    df['SMA_50'] = ta.sma(df['close'], length=50)
    df['SMA_Diff'] = (df['SMA_20'] - df['SMA_50']) / df['close']
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    df['lag_1'] = df['log_ret'].shift(1)
    df['lag_2'] = df['log_ret'].shift(2)
    df['lag_3'] = df['log_ret'].shift(3)
    df['vol_lag'] = df['volume'].shift(1)
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df