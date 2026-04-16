import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from data_loader import get_binance_ohlcv, get_crypto_news
from feature_sentiment import get_sentiment_score
from model import create_cnn_lstm
from simulation import IntradaySimulator

# --- Configuration ---
INITIAL_BALANCE = 1000
SEQ_LENGTH = 4

print("--- STARTING PIPELINE ---")
print(f"Initial Capital: ${INITIAL_BALANCE}")

# --- Step 1: Data Gathering & Logging ---
print("\n[1] Fetching Data...")
df_price = get_binance_ohlcv(symbol='BTC/USDT', timeframe='1h', limit=100)
news_df = get_crypto_news()

# Log News & Sentiment
print(f"   > Fetched {len(news_df)} news headlines.")
headlines = news_df['title'].tolist()
current_sentiment = get_sentiment_score(headlines)
print(f"   > Aggregate Sentiment Score: {current_sentiment:.4f} (0=Bearish, 1=Bullish)")

# Save sentiment context
with open("sentiment_log.txt", "w", encoding='utf-8') as f:
    f.write(f"Aggregate Score: {current_sentiment}\n")
    f.write("Headlines Used:\n")
    for h in headlines[:5]: f.write(f"- {h}\n")
print("   > Sentiment context saved to 'sentiment_log.txt'")

# Map sentiment (Simple fill for demo; ideally time-aligned)
df_price['sentiment'] = current_sentiment 

# --- Step 2: Feature Engineering ---
features = ['open', 'high', 'low', 'close', 'volume', 'sentiment']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_price[features])

def create_sequences(data, seq_length):
    X, y = [], []
    # y_reg for R^2 (next price), y_class for Accuracy (direction)
    y_reg = [] 
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        current_close = data[i + seq_length - 1, 3] # Close is index 3
        next_close = data[i + seq_length, 3]
        
        # Classification Label: 1 if UP, 0 if DOWN
        y.append(1 if next_close > current_close else 0)
        # Regression Label: Actual scaled price
        y_reg.append(next_close)
        
    return np.array(X), np.array(y), np.array(y_reg)

X, y_class, y_reg = create_sequences(scaled_data, seq_length=SEQ_LENGTH)

# Split Data (Train / Val / Test)
# Validation is usually used for model tuning, Test for final simulation
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
y_train = y_class[:train_size]

X_val = X[train_size:train_size+val_size]
y_val = y_class[train_size:train_size+val_size]

X_test = X[train_size+val_size:]
y_test = y_class[train_size+val_size:]
y_reg_test = y_reg[train_size+val_size:] # For R^2 check if model output continuous

print(f"\n[2] Data Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# --- Step 3: Training & Validation Metrics ---
print("\n[3] Training Model...")
model = create_cnn_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
history = model.fit(X_train, y_train, epochs=10, batch_size=4, verbose=0, validation_data=(X_val, y_val))

# Validation Scores
val_preds_prob = model.predict(X_val, verbose=0)
val_preds_bin = [1 if p > 0.5 else 0 for p in val_preds_prob]
val_acc = accuracy_score(y_val, val_preds_bin)
print(f"   > Validation Directional Accuracy: {val_acc*100:.2f}%")

# --- Step 4: Simulation (Test Set) ---
print("\n[4] Running Intraday Simulation (Test Set)...")
simulator = IntradaySimulator(initial_balance=INITIAL_BALANCE)

trade_log = []
test_preds_prob = model.predict(X_test, verbose=0)

# Loop for transparent logging
for i in range(len(test_preds_prob)):
    # Original Data Index mapping
    # Test set starts at: train_size + val_size
    # Sequence offset: + SEQ_LENGTH (lookback)
    # Loop offset: + i
    current_idx_in_df = train_size + val_size + SEQ_LENGTH + i - 1
    
    if current_idx_in_df + 1 >= len(df_price): break
    
    current_price = df_price.iloc[current_idx_in_df]['close']
    next_price = df_price.iloc[current_idx_in_df + 1]['close']
    timestamp = df_price.iloc[current_idx_in_df + 1]['timestamp']
    
    # Model Signal
    prob = test_preds_prob[i][0]
    prediction = 1 if prob > 0.5 else 0
    actual = y_test[i]
    is_correct = (prediction == actual)
    
    # Execute Trade
    old_balance = simulator.balance
    new_balance = simulator.validate_prediction(prob, current_price, next_price)
    action = "HOLD"
    if prob > 0.7: action = "LONG"
    elif prob < 0.3: action = "SHORT"
    
    trade_log.append({
        "Time": timestamp,
        "Price": current_price,
        "Prob": f"{prob:.4f}",
        "Action": action,
        "Correct": is_correct,
        "Balance": f"${new_balance:.2f}"
    })

# --- Step 5: Final Report ---
test_acc = accuracy_score(y_test, [1 if p > 0.5 else 0 for p in test_preds_prob])
print(f"\n[5] Simulation Complete.")
print(f"   > Test Set Directional Accuracy: {test_acc*100:.2f}%")
print(f"   > Final Balance: ${simulator.balance:.2f} (ROI: {((simulator.balance-INITIAL_BALANCE)/INITIAL_BALANCE)*100:.2f}%)")

print("\n--- DETAILED TRADE LOG (Last 5 Actions) ---")
print(f"{'Time':<20} | {'Price':<10} | {'Prob':<8} | {'Action':<6} | {'Correct':<7} | {'Balance':<10}")
for log in trade_log[-5:]:
    print(f"{log['Time']} | {log['Price']:<10.2f} | {log['Prob']} | {log['Action']:<6} | {str(log['Correct']):<7} | {log['Balance']}")

# (Optional) Save full log to CSV
pd.DataFrame(trade_log).to_csv("detailed_trade_log.csv", index=False)
print("\n> Full trade log saved to 'detailed_trade_log.csv'")