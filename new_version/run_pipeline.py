"""
Main Pipeline Orchestrator
══════════════════════════
Step 1: Build feature matrix  (data_engine)
Step 2: Temporal train/val/test split  (NO shuffle)
Step 3: Feature selection on TRAINING ONLY  (feature_selector)
Step 4: Scale features  (fit on train, transform val/test)
Step 5: Build sliding windows + adjacency matrix
Step 6: Train hybrid GCN-BiLSTM
Step 7: Predict on test set
Step 8: Backtest with adjustable position sizing
Step 9: Print report + save artifacts
"""

import os, time, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import config as C
from data_engine import build_features
from feature_selector import run_feature_selection
from hybrid_model import build_temporal_adjacency, create_hybrid_model, get_callbacks
from backtester import Backtester


def temporal_split(df, feature_cols):
    """Temporal train/val/test split. No shuffle."""
    n = len(df)
    train_end = int(n * C.TRAIN_RATIO)
    val_end   = int(n * (C.TRAIN_RATIO + C.VAL_RATIO))

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def create_windows(data: np.ndarray, targets: np.ndarray, window_size: int):
    """Sliding window: X[i] = data[i:i+W], y[i] = target[i+W-1]."""
    X, y = [], []
    for i in range(len(data) - window_size + 1):
        X.append(data[i:i + window_size])
        y.append(targets[i + window_size - 1])
    return np.array(X), np.array(y)


def main():
    t0 = time.time()
    print("=" * 60)
    print("  CRYPTO TRADING PIPELINE — GCN-BiLSTM + Feature Selection")
    print("=" * 60)

    # ── Step 1: Build Features ────────────────────────
    print("\n[1/8] Building feature matrix...")
    df = build_features()
    
    # Separate metadata
    meta_cols  = ["timestamp", "target"]
    price_cols = ["open", "high", "low", "close", "volume"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    print(f"  Total rows: {len(df)}, Total features: {len(feature_cols)}")
    print(f"  Target distribution: {df['target'].value_counts().to_dict()}")

    # ── Step 2: Temporal Split ────────────────────────
    print("\n[2/8] Temporal train/val/test split...")
    train_df, val_df, test_df = temporal_split(df, feature_cols)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ── Step 3: Feature Selection (train only) ────────
    print("\n[3/8] Running feature selection (on training data only)...")
    X_train_raw = train_df[feature_cols]
    y_train_raw = train_df["target"].values

    selected_features = run_feature_selection(X_train_raw, y_train_raw)

    # Ensure close price is always included (needed for backtest)
    for must_have in ["close"]:
        if must_have not in selected_features:
            selected_features.append(must_have)

    print(f"  Selected features ({len(selected_features)}): {selected_features[:10]}...")

    # ── Step 4: Scale (fit on train ONLY) ─────────────
    print("\n[4/8] Scaling features (fit on train, transform val/test)...")
    scaler = StandardScaler()
    
    train_scaled = scaler.fit_transform(train_df[selected_features])
    val_scaled   = scaler.transform(val_df[selected_features])
    test_scaled  = scaler.transform(test_df[selected_features])

    y_train = train_df["target"].values
    y_val   = val_df["target"].values
    y_test  = test_df["target"].values

    # ── Step 5: Sliding Windows + Adjacency ───────────
    print("\n[5/8] Creating sliding windows and graph adjacency...")
    W = C.WINDOW_SIZE

    X_train_w, y_train_w = create_windows(train_scaled, y_train, W)
    X_val_w,   y_val_w   = create_windows(val_scaled,   y_val,   W)
    X_test_w,  y_test_w  = create_windows(test_scaled,  y_test,  W)

    # Adjacency matrix (same for all windows — temporal chain)
    A_norm = build_temporal_adjacency(W)
    
    # Broadcast to batch dimension
    A_train = np.repeat(A_norm[np.newaxis, :, :], len(X_train_w), axis=0)
    A_val   = np.repeat(A_norm[np.newaxis, :, :], len(X_val_w),   axis=0)
    A_test  = np.repeat(A_norm[np.newaxis, :, :], len(X_test_w),  axis=0)

    n_features = X_train_w.shape[2]
    print(f"  Windows — Train: {X_train_w.shape}, Val: {X_val_w.shape}, Test: {X_test_w.shape}")
    print(f"  Adjacency: {A_norm.shape}, Features per step: {n_features}")

    # ── Step 6: Build & Train Model ───────────────────
    print("\n[6/8] Building and training hybrid GCN-BiLSTM model...")
    model = create_hybrid_model(W, n_features)
    model.summary(print_fn=lambda x: None)  # Suppress verbose summary
    
    total_params = model.count_params()
    print(f"  Model parameters: {total_params:,}")

    history = model.fit(
        [X_train_w, A_train], y_train_w,
        validation_data=([X_val_w, A_val], y_val_w),
        epochs=C.EPOCHS,
        batch_size=C.BATCH_SIZE,
        callbacks=get_callbacks(),
        verbose=0
    )

    best_val_loss = min(history.history["val_loss"])
    best_val_acc  = max(history.history["val_accuracy"])
    epochs_run    = len(history.history["loss"])
    print(f"  Training complete in {epochs_run} epochs")
    print(f"  Best val_loss: {best_val_loss:.4f}, Best val_accuracy: {best_val_acc:.4f}")

    # ── Step 7: Predict on Test Set ───────────────────
    print("\n[7/8] Predicting on test set...")
    test_probs = model.predict([X_test_w, A_test], verbose=0).flatten()
    test_preds = (test_probs > 0.5).astype(int)

    acc = accuracy_score(y_test_w, test_preds)
    print(f"  Test Directional Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test_w, test_preds, target_names=["DOWN", "UP"], zero_division=0))

    # ── Step 8: Backtest ──────────────────────────────
    print("[8/8] Running backtest simulation...")
    
    # Map test windows back to original df indices for prices/timestamps
    # Window i in test set corresponds to test_df rows [i : i+W]
    # The prediction is for the LAST row of the window (i + W - 1)
    test_start_idx = len(train_df) + len(val_df)
    
    # Get prices and timestamps for backtest
    # For each prediction i, entry_price = close at (test_start + W - 1 + i)
    # exit_price = close at (test_start + W + i)
    bt_prices = []
    bt_timestamps = []
    bt_actuals = []
    
    for i in range(len(test_probs)):
        idx = test_start_idx + (W - 1) + i
        if idx + 1 < len(df):
            bt_prices.append(df.iloc[idx]["close"])
            bt_timestamps.append(df.iloc[idx]["timestamp"])
            bt_actuals.append(y_test_w[i])

    # Need one extra price for the exit of last trade
    last_idx = test_start_idx + (W - 1) + len(bt_prices)
    if last_idx < len(df):
        bt_prices.append(df.iloc[last_idx]["close"])
    else:
        bt_prices.append(bt_prices[-1])  # Repeat last if at boundary

    bt_prices = np.array(bt_prices)
    bt_actuals = np.array(bt_actuals[:len(bt_prices) - 1])
    bt_probs = test_probs[:len(bt_actuals)]
    bt_timestamps = bt_timestamps[:len(bt_actuals)]

    bt = Backtester(
        initial_capital=C.INITIAL_CAPITAL,
        fee=C.TRADE_FEE,
        long_thresh=C.LONG_THRESHOLD,
        short_thresh=C.SHORT_THRESHOLD,
        position_frac=C.POSITION_FRAC,
    )
    report = bt.run(bt_probs, bt_prices, bt_timestamps, bt_actuals)

    # ── Report ────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  FINAL REPORT")
    print("=" * 60)
    for k, v in report.items():
        label = k.replace("_", " ").title()
        print(f"  {label:<35} {v}")
    print(f"\n  Pipeline time: {elapsed:.1f}s")
    print("=" * 60)

    # Save artifacts
    out_dir = "C:/Users/anurj/Desktop/Capstone/new_version/output"
    os.makedirs(out_dir, exist_ok=True)

    trade_log = bt.get_trade_log()
    trade_log.to_csv(f"{out_dir}/trade_log.csv", index=False)

    eq = bt.get_equity_curve()
    pd.DataFrame({"equity": eq}).to_csv(f"{out_dir}/equity_curve.csv", index=False)

    # Save selected features for reproducibility
    pd.Series(selected_features).to_csv(f"{out_dir}/selected_features.csv", index=False)

    # Save training history
    pd.DataFrame(history.history).to_csv(f"{out_dir}/training_history.csv", index=False)

    print(f"\n  Artifacts saved to {out_dir}/")
    print(f"  - trade_log.csv ({len(trade_log)} rows)")
    print(f"  - equity_curve.csv")
    print(f"  - selected_features.csv")
    print(f"  - training_history.csv")

    # Print last 10 trades for quick inspection
    print("\n  LAST 10 TRADES:")
    print(f"  {'Timestamp':<22} {'Action':<7} {'Prob':>6} {'Entry':>10} {'Exit':>10} {'Correct':>8} {'Capital':>10}")
    for _, row in trade_log.tail(10).iterrows():
        ts_str = str(row["timestamp"])[:19] if row["timestamp"] else "N/A"
        print(f"  {ts_str:<22} {row['action']:<7} {row['prob']:>6.3f} "
              f"{row['entry_price']:>10.2f} {row['exit_price']:>10.2f} "
              f"{str(row['correct']):>8} ${row['capital']:>9.2f}")

    return model, report, trade_log


if __name__ == "__main__":
    model, report, trade_log = main()
