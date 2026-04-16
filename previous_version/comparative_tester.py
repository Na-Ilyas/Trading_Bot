import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

# Local modules
from data_loader import get_processed_data_with_selection
from model import HybridGNNBiLSTM
from simulation import IntradaySimulator

# --- CONFIGURATION ---
SEQ_LENGTH = 24  # Lookback window (24 hours graph)
SEED = 42
INITIAL_CAPITAL = 1000
EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# Set Random Seeds
np.random.seed(SEED)
torch.manual_seed(SEED)

print("--- STARTING PYTORCH HYBRID GNN-BiLSTM EXPERIMENT ---")

# --- 1. DATA PREPARATION ---
print("\n[1] Preparing Data...")
scaled_data, raw_df, scaler, selected_features = get_processed_data_with_selection()

def create_sequences(data, raw_prices, seq_length):
    X, y, indices = [], [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        # Target
        current_close = raw_prices.iloc[i + seq_length - 1]['close']
        next_close = raw_prices.iloc[i + seq_length]['close']
        y.append(1 if next_close > current_close else 0)
        indices.append(i + seq_length - 1)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(indices)

X, y, idx_map = create_sequences(scaled_data, raw_df, SEQ_LENGTH)

# Train/Test Split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
idx_test = idx_map[split_idx:]

print(f"   > Feature Shape: {X_train.shape} | Features: {len(selected_features)}")

# --- 2. HELPER FUNCTIONS ---

def train_model(X_data, y_data, class_weights=None):
    """Generic PyTorch Training Loop"""
    # Create Dataset
    dataset = TensorDataset(torch.from_numpy(X_data), torch.from_numpy(y_data))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Init Model
    model = HybridGNNBiLSTM(input_dim=X_data.shape[2], seq_length=SEQ_LENGTH).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Loss Function (Handles Class Imbalance via pos_weight)
    pos_weight = None
    if class_weights:
        # pos_weight = count(neg) / count(pos) roughly equal to weight[1] / weight[0]
        p_w = class_weights[1] / class_weights[0]
        pos_weight = torch.tensor([p_w]).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            logits = model(xb)
            loss = criterion(logits.squeeze(), yb)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    return model

def get_predictions(model, X_data):
    """Inference Loop"""
    model.eval()
    with torch.no_grad():
        tensor_X = torch.from_numpy(X_data).to(device)
        logits = model(tensor_X)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    return probs

# --- 3. MODEL A: UNDERSAMPLING ---
print("\n[2] Training Model A (Undersampling)...")
idx_0 = np.where(y_train == 0)[0]
idx_1 = np.where(y_train == 1)[0]
min_len = min(len(idx_0), len(idx_1))

# Shuffle and Slice
np.random.shuffle(idx_0); np.random.shuffle(idx_1)
idx_bal = np.concatenate([idx_0[:min_len], idx_1[:min_len]])
np.random.shuffle(idx_bal)

X_train_A = X_train[idx_bal]
y_train_A = y_train[idx_bal]

model_a = train_model(X_train_A, y_train_A, class_weights=None)
probs_A = get_predictions(model_a, X_test)

# --- 4. MODEL B: CLASS WEIGHTING ---
print("\n[3] Training Model B (Class Weighting)...")
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
w_dict = {0: weights[0], 1: weights[1]}
print(f"   > Weights: {w_dict}")

model_b = train_model(X_train, y_train, class_weights=w_dict)
probs_B = get_predictions(model_b, X_test)

# --- 5. DYNAMIC THRESHOLDS & SIMULATION ---
print("\n[4] Simulating with Dynamic Thresholds...")

def simulate_strategy(probs, name):
    sim = IntradaySimulator(initial_balance=INITIAL_CAPITAL)
    
    # Calculate Dynamic Thresholds
    mean_p, std_p = np.mean(probs), np.std(probs)
    std_p = max(std_p, 0.02) # Minimum volatility floor
    LONG = mean_p + (0.5 * std_p)
    SHORT = mean_p - (0.5 * std_p)
    
    print(f"   > {name} Stats: Mean={mean_p:.3f} | L>{LONG:.3f}, S<{SHORT:.3f}")
    
    trade_count = 0
    
    for i in range(len(probs)):
        idx = idx_test[i]
        if idx + 1 >= len(raw_df): break
        
        curr_p = raw_df.iloc[idx]['close']
        next_p = raw_df.iloc[idx+1]['close']
        
        p = probs[i]
        sim.validate_prediction(p, curr_p, next_p, LONG, SHORT)
        
        if p > LONG or p < SHORT: trade_count += 1
            
    return sim.balance, trade_count

bal_A, trades_A = simulate_strategy(probs_A, "Model A")
bal_B, trades_B = simulate_strategy(probs_B, "Model B")

# --- 6. REPORT ---
acc_A = accuracy_score(y_test, [1 if p > 0.5 else 0 for p in probs_A])
acc_B = accuracy_score(y_test, [1 if p > 0.5 else 0 for p in probs_B])

print("\n" + "="*60)
print(f"{'METRIC':<20} | {'MODEL A (Undersample)':<22} | {'MODEL B (Weighted)':<20}")
print("-" * 60)
print(f"{'Accuracy':<20} | {acc_A*100:.2f}%{'':<16} | {acc_B*100:.2f}%")
print(f"{'Final Balance':<20} | ${bal_A:.2f}{'':<15} | ${bal_B:.2f}")
print(f"{'Trades Executed':<20} | {trades_A:<22} | {trades_B}")
print("="*60)