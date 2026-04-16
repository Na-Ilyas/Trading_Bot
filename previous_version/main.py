import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import logging
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
# Local modules (Ensure these exist in your directory)
from data_api import fetch_historical_data
from data_loader import feature_engineering
from model import HybridGNNBiLSTM
from simulation import IntradaySimulator

# --- CONFIGURATION ---
SEQ_LENGTH = 24
INITIAL_CAPITAL = 1000
EPOCHS = 50
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
SEED = 42
FEATURE_K = 10 # Number of top features to select

# Setup Device & Logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seeds(seed):
    """Ensures reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_sequences(data, targets, seq_length):
    """Windows the time-series data into 3D tensors for LSTM/GNN."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        # Target aligns with the end of the sequence predicting the NEXT candle
        y.append(targets[i + seq_length - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_model_with_early_stopping(X_train, y_train):
    """Trains the GNN-BiLSTM with Undersampling and Best-Weight Checkpointing."""
    # 1. Undersampling to balance the dataset (Noise Reduction)
    idx_0 = np.where(y_train == 0)[0]
    idx_1 = np.where(y_train == 1)[0]
    min_len = min(len(idx_0), len(idx_1))
    
    np.random.shuffle(idx_0); np.random.shuffle(idx_1)
    idx_bal = np.concatenate([idx_0[:min_len], idx_1[:min_len]])
    np.random.shuffle(idx_bal)
    
    X_bal, y_bal = X_train[idx_bal], y_train[idx_bal]
    
    dataset = TensorDataset(torch.from_numpy(X_bal), torch.from_numpy(y_bal))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Initialize Model
    model = HybridGNNBiLSTM(num_features=X_train.shape[2], seq_length=SEQ_LENGTH).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # 3. Early Stopping & Checkpointing Logic
    best_loss = float('inf')
    best_model_weights = None
    patience, patience_counter = 7, 0
    
    logging.info(f"Starting Training on {len(X_bal)} balanced samples...")
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
            
        avg_loss = epoch_loss / len(loader)
        
        # Checkpoint if model improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            logging.info(f"   Epoch {epoch+1:02d}: Loss improved to {avg_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.warning(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    # Restore the best weights before returning (Crucial for production!)
    logging.info(f"Restoring best model weights (Loss: {best_loss:.4f})")
    model.load_state_dict(best_model_weights)
    return model

if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Previous Version Pipeline")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date DD.MM.YYYY")
    parser.add_argument("--end", type=str, default=None,
                        help="End date DD.MM.YYYY")
    parser.add_argument("--input", type=int, default=3000,
                        help="Number of candles (default 3000)")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%d.%m.%Y") if args.start else None
    end_date = datetime.strptime(args.end, "%d.%m.%Y") if args.end else None

    set_seeds(SEED)
    logging.info("=== STARTING PRODUCTION ML PIPELINE ===")

    # ---------------------------------------------------------
    # STEP 1: Data Ingestion & Feature Engineering
    # ---------------------------------------------------------
    raw_df = fetch_historical_data(total_candles=args.input,
                                   start_date=start_date, end_date=end_date)
    if raw_df.empty:
        logging.error("Failed to fetch data. Exiting.")
        exit(1)
        
    df = feature_engineering(raw_df)
    
    # Exclude non-features
    exclude_cols = ['timestamp', 'target', 'open', 'high', 'low', 'close']
    candidate_features = [c for c in df.columns if c not in exclude_cols]
    
    # ---------------------------------------------------------
    # STEP 2: Strict Time-Based Split (No Look-Ahead Bias)
    # ---------------------------------------------------------
    split_idx = int(len(df) * 0.9)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    logging.info(f"Data Split -> Train: {len(train_df)} hrs | Test: {len(test_df)} hrs")

   # STEP 3: Feature Selection (Fitted on Train ONLY)
    # ---------------------------------------------------------
    logging.info("Running Boruta Feature Selection on Train Data...")
    

    X_train_features = train_df[candidate_features].values
    y_train_target = train_df['target'].values
    
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    
    try:
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, max_iter=50)
        feat_selector.fit(X_train_features, y_train_target)
        
        selected_indices = np.where(feat_selector.support_)[0]
        selected_features = [candidate_features[i] for i in selected_indices]
        
        if len(selected_features) == 0:
            raise ValueError("Boruta selected 0 features.")
        logging.info(f"Boruta selected: {selected_features}")
        
    except Exception as e:
        logging.warning(f"Boruta Failed: {e}. Falling back to SelectKBest.")
        selector = SelectKBest(f_classif, k=min(10, len(candidate_features)))
        selector.fit(X_train_features, y_train_target)
        selected_indices = selector.get_support(indices=True)
        selected_features = [candidate_features[i] for i in selected_indices]
        logging.info(f"SelectKBest selected: {selected_features}")
    
    # ---------------------------------------------------------
    # STEP 4: Scaling (Fitted on Train ONLY)
    # ---------------------------------------------------------
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[selected_features])
    test_scaled = scaler.transform(test_df[selected_features])
    
    # ---------------------------------------------------------
    # STEP 5: Sequence Generation
    # ---------------------------------------------------------
    X_train, y_train = create_sequences(train_scaled, train_df['target'].values, SEQ_LENGTH)
    X_test, y_test = create_sequences(test_scaled, test_df['target'].values, SEQ_LENGTH)
    
    # ---------------------------------------------------------
    # STEP 6: Model Training
    # ---------------------------------------------------------
    model = train_model_with_early_stopping(X_train, y_train)
    
    # ---------------------------------------------------------
    # STEP 7: Backtest / Validation on Unseen Data
    # ---------------------------------------------------------
    logging.info("\n=== RUNNING BACKTEST ON OUT-OF-SAMPLE DATA ===")
    model.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(X_test).to(device)
        logits = model(test_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
    # Dynamic Threshold Calculation based on actual model variance
    mean_p, std_p = np.mean(probs), np.std(probs)
    LONG_THRESH = mean_p + (0.5 * std_p)
    SHORT_THRESH = mean_p - (0.5 * std_p)
    
    sim = IntradaySimulator(initial_balance=INITIAL_CAPITAL)
    price_series = test_df['close'].values
    
    # Run historical execution loop
    for i in range(len(probs)):
        curr_price = price_series[i + SEQ_LENGTH - 1]
        next_price = price_series[i + SEQ_LENGTH]
        
        sim.validate_prediction(probs[i], curr_price, next_price, LONG_THRESH, SHORT_THRESH)
        
    # Final Metrics
    acc = accuracy_score(y_test, [1 if p > 0.5 else 0 for p in probs])
    roi = ((sim.balance - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    print("\n" + "="*50)
    print(probs[:10]) # Print first 10 probabilities for sanity check
    print("="*50)
    logging.info("\n" + "="*50)
    logging.info(f"BACKTEST RESULTS (Test Set)")
    logging.info("="*50)
    logging.info(f"Directional Accuracy : {acc*100:.2f}%")
    logging.info(f"Total Trades Executed: {sim.trade_count}")
    logging.info(f"Final Account Balance: ${sim.balance:.2f} ({roi:+.2f}%)")
    logging.info("="*50)

    # ---------------------------------------------------------
    # STEP 8: Export Production Artifacts for Live Bot
    # ---------------------------------------------------------
    logging.info("\nSaving Production Artifacts...")
    os.makedirs("production_assets", exist_ok=True)
    
    # 1. Save Model Weights
    torch.save(model.state_dict(), "production_assets/hybrid_model.pth")
    # 2. Save Scaler
    joblib.dump(scaler, "production_assets/scaler.pkl")
    # 3. Save Selected Features list
    joblib.dump(selected_features, "production_assets/features.pkl")
    
    logging.info("SUCCESS: Model and artifacts saved to '/production_assets/'. Ready for live trading.")