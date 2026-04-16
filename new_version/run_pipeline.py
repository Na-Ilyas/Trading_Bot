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
Step 9: Print report + save artifacts + charts

Usage:
  python run_pipeline.py                                  # default (synthetic 3000 candles)
  python run_pipeline.py --input 4000                     # custom candle count (last 4000 hours)
  python run_pipeline.py --start 06.10.2025 --end 06.02.2026  # custom date range
  python run_pipeline.py --compare                        # compare all models
  python run_pipeline.py --start 06.10.2025 --end 06.02.2026 --compare
"""

import os, sys, time, json, argparse, random, warnings
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = "42"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf

import config as C


def set_global_seeds(seed=C.SYNTHETIC_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from data_engine import build_features
from feature_selector import run_feature_selection
from hybrid_model import build_temporal_adjacency, create_hybrid_model, get_callbacks
from backtester import Backtester


# ══════════════════════════════════════════════════════════
#  CLI Argument Parsing
# ══════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Crypto Trading Pipeline — GCN-BiLSTM + Feature Selection"
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date in DD.MM.YYYY format (e.g. 06.10.2025)"
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date in DD.MM.YYYY format (e.g. 06.02.2026)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run comparative mode: train all models on the same data"
    )
    parser.add_argument(
        "--input", type=int, default=None,
        help="Number of hours/candles for synthetic data (default: 3000)"
    )
    args = parser.parse_args()

    start_date = None
    end_date = None
    if args.start:
        start_date = datetime.strptime(args.start, "%d.%m.%Y")
    if args.end:
        end_date = datetime.strptime(args.end, "%d.%m.%Y")

    if (start_date is None) != (end_date is None):
        parser.error("Both --start and --end must be provided together")

    if start_date and end_date and start_date >= end_date:
        parser.error("--start must be before --end")

    if args.input is not None and args.input < 100:
        parser.error("--input must be at least 100")

    if args.input is not None and (args.start or args.end):
        parser.error("--input cannot be used with --start/--end (date range determines data size)")

    n_candles = args.input if args.input else C.N_CANDLES

    return start_date, end_date, args.compare, n_candles


# ══════════════════════════════════════════════════════════
#  Helper Functions
# ══════════════════════════════════════════════════════════
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


def calibrate_thresholds(val_probs: np.ndarray, target_trade_pct: float = 0.40):
    """
    Calibrate LONG/SHORT thresholds from validation predictions so that
    approximately target_trade_pct of signals result in trades.
    Uses percentiles of the probability distribution.
    """
    long_thresh = np.percentile(val_probs, (1 - target_trade_pct / 2) * 100)
    short_thresh = np.percentile(val_probs, (target_trade_pct / 2) * 100)
    # Ensure thresholds are meaningful (long > 0.5, short < 0.5)
    long_thresh = max(long_thresh, 0.505)
    short_thresh = min(short_thresh, 0.495)
    return float(long_thresh), float(short_thresh)


def build_output_dir(start_date, end_date, compare_mode, n_candles):
    """
    Build output directory path with descriptive naming and dedup suffix.

    Naming:
      default:    run_{YYYYMMDD}_last_{N}
      date range: run_{YYYYMMDD}_{DDMMYYYY}_to_{DDMMYYYY}
      compare:    append _compare

    If the directory already exists, append _1, _2, etc.
    """
    base_path = "C:/Users/anurj/Desktop/Capstone/new_version/output"
    today = datetime.now().strftime("%Y%m%d")

    if start_date and end_date:
        s = start_date.strftime("%d%m%Y")
        e = end_date.strftime("%d%m%Y")
        name = f"run_{today}_{s}_to_{e}"
    else:
        name = f"run_{today}_last_{n_candles}"

    if compare_mode:
        name += "_compare"

    out_dir = f"{base_path}/{name}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    # Dedup: try _1, _2, ...
    counter = 1
    while True:
        candidate = f"{base_path}/{name}_{counter}"
        if not os.path.exists(candidate):
            os.makedirs(candidate, exist_ok=True)
            return candidate
        counter += 1


def prepare_backtest_data(df, train_df, val_df, test_probs, y_test_w, W):
    """Map test windows back to original df indices for prices/timestamps."""
    test_start_idx = len(train_df) + len(val_df)

    bt_prices = []
    bt_timestamps = []
    bt_actuals = []

    for i in range(len(test_probs)):
        idx = test_start_idx + (W - 1) + i
        if idx + 1 < len(df):
            bt_prices.append(df.iloc[idx]["close"])
            bt_timestamps.append(df.iloc[idx]["timestamp"])
            bt_actuals.append(y_test_w[i])

    last_idx = test_start_idx + (W - 1) + len(bt_prices)
    if last_idx < len(df):
        bt_prices.append(df.iloc[last_idx]["close"])
    else:
        bt_prices.append(bt_prices[-1])

    bt_prices = np.array(bt_prices)
    bt_actuals = np.array(bt_actuals[:len(bt_prices) - 1])
    bt_probs = test_probs[:len(bt_actuals)]
    bt_timestamps = bt_timestamps[:len(bt_actuals)]

    return bt_probs, bt_prices, bt_timestamps, bt_actuals


# ══════════════════════════════════════════════════════════
#  Chart Generation
# ══════════════════════════════════════════════════════════
def generate_charts(out_dir, equity_curve, trade_log_df, history_dict):
    """Generate and save matplotlib charts."""

    # Closed trades (completed positions) for PnL analysis
    closed = trade_log_df[trade_log_df["event_type"].isin(["CLOSE", "FORCE_CLOSE"])]

    # 1. Equity Curve with Drawdown
    fig, ax1 = plt.subplots(figsize=(14, 5))
    eq = np.array(equity_curve)
    ax1.plot(eq, color="steelblue", linewidth=1.2, label="Equity (mark-to-market)")

    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / (running_max + 1e-10) * 100
    ax2 = ax1.twinx()
    ax2.fill_between(range(len(dd)), dd, 0, alpha=0.2, color="red", label="Drawdown %")
    ax2.set_ylabel("Drawdown %", color="red")

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Capital ($)", color="steelblue")
    ax1.set_title("Equity Curve & Drawdown")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/equity_curve.png", dpi=150)
    plt.close()

    # 2. PnL Distribution (% and $)
    if len(closed) > 0:
        fig, (ax_pct, ax_amt) = plt.subplots(1, 2, figsize=(14, 5))

        pnl_vals = closed["pnl_pct"].values
        ax_pct.hist(pnl_vals, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        ax_pct.axvline(0, color="red", linestyle="--", linewidth=1)
        ax_pct.axvline(pnl_vals.mean(), color="green", linestyle="--", linewidth=1,
                       label=f"Mean: {pnl_vals.mean():.2f}%")
        ax_pct.set_xlabel("Position PnL %")
        ax_pct.set_ylabel("Count")
        ax_pct.set_title("Closed Position PnL Distribution (%)")
        ax_pct.legend()

        pnl_amts = closed["pnl_amount"].values
        ax_amt.hist(pnl_amts, bins=30, color="darkgreen", edgecolor="black", alpha=0.7)
        ax_amt.axvline(0, color="red", linestyle="--", linewidth=1)
        ax_amt.axvline(pnl_amts.mean(), color="orange", linestyle="--", linewidth=1,
                       label=f"Mean: ${pnl_amts.mean():.2f}")
        ax_amt.set_xlabel("Position PnL ($)")
        ax_amt.set_ylabel("Count")
        ax_amt.set_title("Closed Position PnL Distribution ($)")
        ax_amt.legend()

        plt.tight_layout()
        plt.savefig(f"{out_dir}/pnl_distribution.png", dpi=150)
        plt.close()

    # 3. Training Curves
    if history_dict:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(history_dict.get("loss", []), label="Train Loss", color="steelblue")
        ax1.plot(history_dict.get("val_loss", []), label="Val Loss", color="coral")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()

        ax2.plot(history_dict.get("accuracy", []), label="Train Acc", color="steelblue")
        ax2.plot(history_dict.get("val_accuracy", []), label="Val Acc", color="coral")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training & Validation Accuracy")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"{out_dir}/training_curves.png", dpi=150)
        plt.close()

    # 4. Cumulative PnL over time
    if len(closed) > 0:
        fig, ax = plt.subplots(figsize=(14, 5))
        cum_pnl = trade_log_df["cumulative_pnl"].values
        ax.plot(cum_pnl, color="steelblue", linewidth=1.2)
        ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                        where=(np.array(cum_pnl) >= 0), color="green", alpha=0.15)
        ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                        where=(np.array(cum_pnl) < 0), color="red", alpha=0.15)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative PnL ($)")
        ax.set_title("Cumulative Profit & Loss Over Time")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/cumulative_pnl.png", dpi=150)
        plt.close()

    # 5. Prediction Confidence Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    probs = trade_log_df["prob"].values
    colors = ["green" if a == "LONG" else "red" if a == "SHORT" else "gray"
              for a in trade_log_df["action"].values]
    ax.scatter(range(len(probs)), probs, c=colors, s=6, alpha=0.6)
    ax.axhline(0.5, color="black", linestyle="-", linewidth=0.8, label="Neutral (0.5)")
    unique_actions = trade_log_df["action"].unique()
    if "LONG" in unique_actions:
        long_probs = trade_log_df[trade_log_df["action"] == "LONG"]["prob"]
        ax.axhline(long_probs.min(), color="green", linestyle="--", linewidth=0.8,
                   alpha=0.7, label=f"Long thresh ({long_probs.min():.4f})")
    if "SHORT" in unique_actions:
        short_probs = trade_log_df[trade_log_df["action"] == "SHORT"]["prob"]
        ax.axhline(short_probs.max(), color="red", linestyle="--", linewidth=0.8,
                   alpha=0.7, label=f"Short thresh ({short_probs.max():.4f})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Model Probability P(up)")
    ax.set_title("Model Confidence Over Time (green=LONG, red=SHORT, gray=HOLD)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/prediction_confidence.png", dpi=150)
    plt.close()

    # 6. Win Rate Over Time (rolling) — based on closed positions
    if len(closed) > 0 and len(closed) >= 10:
        fig, ax = plt.subplots(figsize=(14, 5))
        win_series = (closed["pnl_pct"] > 0).astype(float).reset_index(drop=True)
        rolling_wr = win_series.rolling(window=min(20, len(closed)), min_periods=5).mean() * 100
        ax.plot(rolling_wr.values, color="steelblue", linewidth=1.2, label="Rolling Win Rate (20)")
        ax.axhline(50, color="red", linestyle="--", linewidth=0.8, label="50% Baseline")
        ax.set_xlabel("Closed Position #")
        ax.set_ylabel("Win Rate %")
        ax.set_title("Rolling Win Rate (20-position window)")
        ax.legend()
        ax.set_ylim(0, 100)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/rolling_win_rate.png", dpi=150)
        plt.close()

    # 7. Signal Distribution Pie Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    action_counts = trade_log_df["action"].value_counts()
    colors_pie = {"LONG": "#2ecc71", "SHORT": "#e74c3c", "HOLD": "#95a5a6", "FORCE_CLOSE": "#3498db"}
    ax1.pie(action_counts.values,
            labels=action_counts.index,
            colors=[colors_pie.get(a, "gray") for a in action_counts.index],
            autopct="%1.1f%%", startangle=90)
    ax1.set_title("Signal Distribution")

    if len(closed) > 0:
        wins = (closed["pnl_pct"] > 0).sum()
        losses = (closed["pnl_pct"] <= 0).sum()
        ax2.pie([wins, losses], labels=["Win", "Loss"],
                colors=["#2ecc71", "#e74c3c"],
                autopct="%1.1f%%", startangle=90)
        ax2.set_title(f"Win / Loss ({len(closed)} closed positions)")
    else:
        ax2.text(0.5, 0.5, "No trades", ha="center", va="center", fontsize=14)
        ax2.set_title("Win / Loss Ratio")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/action_breakdown.png", dpi=150)
    plt.close()


def generate_comparison_chart(out_dir, comparison_df):
    """Generate grouped bar chart comparing models."""
    metrics = ["Test Acc %", "Trade Acc %", "Return %", "Sharpe", "Max DD %", "Win Rate %"]
    available = [m for m in metrics if m in comparison_df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, len(comparison_df)))

    for i, metric in enumerate(available):
        ax = axes[i]
        bars = ax.bar(comparison_df["Model"], comparison_df[metric], color=colors)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel(metric)
        for bar, val in zip(bars, comparison_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", rotation=30)

    # Hide unused subplots
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/model_comparison.png", dpi=150)
    plt.close()


def save_run_summary(out_dir, report, args_info, model_params, epochs_run,
                     best_val_loss, best_val_acc, n_features, data_shape,
                     calibrated_long=None, calibrated_short=None):
    """Save comprehensive run summary as JSON."""
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "date_range": args_info,
        "data": {
            "total_rows": data_shape[0],
            "total_features_raw": data_shape[1],
            "features_selected": n_features,
        },
        "model": {
            "total_params": model_params,
            "epochs_run": epochs_run,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_acc,
        },
        "backtest": report,
        "config_snapshot": {
            "window_size": C.WINDOW_SIZE,
            "train_ratio": C.TRAIN_RATIO,
            "val_ratio": C.VAL_RATIO,
            "test_ratio": C.TEST_RATIO,
            "batch_size": C.BATCH_SIZE,
            "learning_rate": C.LEARNING_RATE,
            "dropout_rate": C.DROPOUT_RATE,
            "long_threshold_config": C.LONG_THRESHOLD,
            "short_threshold_config": C.SHORT_THRESHOLD,
            "calibrated_long_threshold": calibrated_long,
            "calibrated_short_threshold": calibrated_short,
            "position_frac": C.POSITION_FRAC,
            "use_kelly": C.USE_KELLY,
            "use_attention": C.USE_ATTENTION,
            "use_residual": C.USE_RESIDUAL,
            "label_smoothing": C.LABEL_SMOOTHING,
            "use_learnable_adj": C.USE_LEARNABLE_ADJ,
        },
    }
    with open(f"{out_dir}/run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


# ══════════════════════════════════════════════════════════
#  Comparative Mode
# ══════════════════════════════════════════════════════════
def _build_compare_row(model_name, test_acc, report, train_time):
    """Build a full result row from a backtester report for comparative CSV."""
    return {
        "Model": model_name,
        "Test Acc %": round(test_acc, 2),
        "Trade Acc %": report["directional_accuracy_trades"],
        "Overall Acc %": report["directional_accuracy_all"],
        "Return %": report["total_return_pct"],
        "Sharpe": report["sharpe_ratio"],
        "Max DD %": report["max_drawdown_pct"],
        "Win Rate %": report["win_rate_pct"],
        "Trades": report["total_trades"],
        "Longs": report["n_longs"],
        "Shorts": report["n_shorts"],
        "Holds": report["n_holds"],
        "Initial Capital": report["initial_capital"],
        "Final Capital": report["final_capital"],
        "Max Equity": report["max_equity"],
        "Profit Factor": report["profit_factor"],
        "Avg Trade PnL %": report["avg_trade_pnl_pct"],
        "Biggest Win %": report["biggest_win_pct"],
        "Biggest Loss %": report["biggest_loss_pct"],
        "Biggest Trade $": report["biggest_trade_amount"],
        "Total Fees $": report["total_fees_paid"],
        "Max Win Streak": report["max_win_streak"],
        "Max Loss Streak": report["max_loss_streak"],
        "Positions Opened": report.get("n_positions_opened", report["total_trades"]),
        "Avg Duration (h)": report.get("avg_position_duration_hours", 0.0),
        "Train Time (s)": round(train_time, 1),
    }


def _save_model_artifacts(out_dir, model_name, bt):
    """Save trade log and equity curve for a single model in compare mode."""
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    model_dir = f"{out_dir}/{safe_name}"
    os.makedirs(model_dir, exist_ok=True)

    trade_log = bt.get_trade_log()
    trade_log.to_csv(f"{model_dir}/trade_log.csv", index=False)

    eq = bt.get_equity_curve()
    pd.DataFrame({"equity": eq}).to_csv(f"{model_dir}/equity_curve.csv", index=False)


def run_comparative(df, train_df, val_df, test_df, selected_features,
                    X_train_w, y_train_w, X_val_w, y_val_w,
                    X_test_w, y_test_w, A_train, A_val, A_test,
                    out_dir, W, date_info=""):
    """Train all models on the same data and compare."""
    from baseline_models import get_all_baselines

    results = []
    n_features = X_train_w.shape[2]

    # 1. Hybrid GCN-BiLSTM
    print("\n  [1/6] Training Hybrid GCN-BiLSTM...")
    t0 = time.time()
    hybrid = create_hybrid_model(W, n_features)
    history = hybrid.fit(
        [X_train_w, A_train], y_train_w,
        validation_data=([X_val_w, A_val], y_val_w),
        epochs=C.EPOCHS, batch_size=C.BATCH_SIZE,
        callbacks=get_callbacks(), verbose=0
    )
    hybrid_time = time.time() - t0
    hybrid_probs = hybrid.predict([X_test_w, A_test], verbose=0).flatten()
    hybrid_preds = (hybrid_probs > 0.5).astype(int)
    hybrid_acc = accuracy_score(y_test_w, hybrid_preds) * 100

    # Calibrate thresholds from validation predictions
    hybrid_val_probs = hybrid.predict([X_val_w, A_val], verbose=0).flatten()
    h_long, h_short = calibrate_thresholds(hybrid_val_probs)

    # Backtest hybrid
    bt_probs, bt_prices, bt_ts, bt_actuals = prepare_backtest_data(
        df, train_df, val_df, hybrid_probs, y_test_w, W)
    bt = Backtester(long_thresh=h_long, short_thresh=h_short)
    hybrid_report = bt.run(bt_probs, bt_prices, bt_ts, bt_actuals)

    results.append(_build_compare_row("Hybrid (GCN-BiLSTM)", hybrid_acc, hybrid_report, hybrid_time))
    _save_model_artifacts(out_dir, "Hybrid (GCN-BiLSTM)", bt)

    # 2-6. Baseline models
    baselines = get_all_baselines()
    for i, model in enumerate(baselines):
        print(f"  [{i+2}/6] Training {model.name}...")
        t0 = time.time()

        if model.name == "ARIMA":
            # ARIMA uses raw close prices
            model.fit(X_train_w, y_train_w)
            train_close = train_df["close"].values
            probs = model.predict_from_prices(train_close, len(y_test_w))
            val_probs_m = np.full(len(y_val_w), 0.5)  # ARIMA has no val predictions
            train_time = time.time() - t0
        elif model.name == "BuyAndHold":
            info = model.fit(X_train_w, y_train_w)
            train_time = info.get("train_time", 0.0)
            probs = model.predict(X_test_w)
            val_probs_m = model.predict(X_val_w)
        else:
            info = model.fit(X_train_w, y_train_w, X_val_w, y_val_w)
            train_time = info.get("train_time", time.time() - t0)
            probs = model.predict(X_test_w)
            val_probs_m = model.predict(X_val_w)

        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(y_test_w, preds) * 100

        # Calibrate thresholds per model
        m_long, m_short = calibrate_thresholds(val_probs_m)

        # Backtest
        bt_probs_m, bt_prices_m, bt_ts_m, bt_actuals_m = prepare_backtest_data(
            df, train_df, val_df, probs, y_test_w, W)
        bt_m = Backtester(long_thresh=m_long, short_thresh=m_short)
        report_m = bt_m.run(bt_probs_m, bt_prices_m, bt_ts_m, bt_actuals_m)

        results.append(_build_compare_row(model.name, acc, report_m, train_time))
        _save_model_artifacts(out_dir, model.name, bt_m)

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(f"{out_dir}/comparative_results.csv", index=False)

    generate_comparison_chart(out_dir, comparison_df)

    # Save compare summary JSON
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "date_range": date_info,
        "mode": "comparative",
        "models_compared": len(results),
        "data": {
            "total_rows": len(df),
            "features_selected": len(selected_features),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
        },
        "results": results,
        "config_snapshot": {
            "window_size": C.WINDOW_SIZE,
            "train_ratio": C.TRAIN_RATIO,
            "val_ratio": C.VAL_RATIO,
            "test_ratio": C.TEST_RATIO,
            "batch_size": C.BATCH_SIZE,
            "learning_rate": C.LEARNING_RATE,
            "dropout_rate": C.DROPOUT_RATE,
            "position_frac": C.POSITION_FRAC,
            "use_kelly": C.USE_KELLY,
        },
    }
    with open(f"{out_dir}/compare_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return comparison_df


# ══════════════════════════════════════════════════════════
#  Main Pipeline
# ══════════════════════════════════════════════════════════
def main():
    set_global_seeds()
    start_date, end_date, compare_mode, n_candles = parse_args()
    t0 = time.time()

    # Date range info string
    if start_date and end_date:
        date_info = f"{start_date.strftime('%d.%m.%Y')} to {end_date.strftime('%d.%m.%Y')}"
    else:
        date_info = f"Last {n_candles} candles (synthetic)"

    print("=" * 60)
    print("  CRYPTO TRADING PIPELINE — GCN-BiLSTM + Feature Selection")
    print(f"  Date range: {date_info}")
    if compare_mode:
        print("  Mode: COMPARATIVE (all models)")
    print("=" * 60)

    # ── Step 1: Build Features ────────────────────────
    print("\n[1/8] Building feature matrix...")
    df = build_features(start_date=start_date, end_date=end_date, n_candles=n_candles)

    meta_cols  = ["timestamp", "target"]
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

    A_norm = build_temporal_adjacency(W)
    A_train = np.repeat(A_norm[np.newaxis, :, :], len(X_train_w), axis=0)
    A_val   = np.repeat(A_norm[np.newaxis, :, :], len(X_val_w),   axis=0)
    A_test  = np.repeat(A_norm[np.newaxis, :, :], len(X_test_w),  axis=0)

    n_features = X_train_w.shape[2]
    print(f"  Windows — Train: {X_train_w.shape}, Val: {X_val_w.shape}, Test: {X_test_w.shape}")
    print(f"  Adjacency: {A_norm.shape}, Features per step: {n_features}")

    # ── Setup output directory ────────────────────────
    out_dir = build_output_dir(start_date, end_date, compare_mode, n_candles)

    # ══════════════════════════════════════════════════
    #  COMPARATIVE MODE
    # ══════════════════════════════════════════════════
    if compare_mode:
        print("\n[6/8] Running comparative analysis...")
        comparison_df = run_comparative(
            df, train_df, val_df, test_df, selected_features,
            X_train_w, y_train_w, X_val_w, y_val_w,
            X_test_w, y_test_w, A_train, A_val, A_test,
            out_dir, W, date_info=date_info
        )

        elapsed = time.time() - t0
        print("\n" + "=" * 60)
        print("  COMPARATIVE RESULTS")
        print("=" * 60)
        # Print key columns to console (full data is in CSV)
        display_cols = ["Model", "Return %", "Sharpe", "Max DD %", "Win Rate %",
                        "Trades", "Initial Capital", "Final Capital",
                        "Biggest Win %", "Biggest Loss %"]
        print(comparison_df[display_cols].to_string(index=False))
        print(f"\n  Pipeline time: {elapsed:.1f}s")
        print(f"  Results saved to {out_dir}/")
        print(f"  - comparative_results.csv ({len(comparison_df.columns)} columns)")
        print(f"  - compare_summary.json")
        print(f"  - model_comparison.png")
        print(f"  - Per-model subdirectories with trade_log.csv & equity_curve.csv")
        print("=" * 60)

        # Save features for reproducibility
        pd.Series(selected_features).to_csv(f"{out_dir}/selected_features.csv", index=False)

        return None, comparison_df, None

    # ══════════════════════════════════════════════════
    #  SINGLE MODEL MODE (default)
    # ══════════════════════════════════════════════════

    # ── Step 6: Build & Train Model ───────────────────
    print("\n[6/8] Building and training hybrid GCN-BiLSTM model...")
    model = create_hybrid_model(W, n_features)
    model.summary(print_fn=lambda x: None)

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

    # Calibrate thresholds from validation predictions
    val_probs = model.predict([X_val_w, A_val], verbose=0).flatten()
    long_thresh, short_thresh = calibrate_thresholds(val_probs)
    print(f"  Calibrated thresholds — LONG: {long_thresh:.4f}, SHORT: {short_thresh:.4f}")

    # ── Step 8: Backtest ──────────────────────────────
    print("[8/8] Running backtest simulation...")

    bt_probs, bt_prices, bt_timestamps, bt_actuals = prepare_backtest_data(
        df, train_df, val_df, test_probs, y_test_w, W)

    bt = Backtester(
        initial_capital=C.INITIAL_CAPITAL,
        fee=C.TRADE_FEE,
        long_thresh=long_thresh,
        short_thresh=short_thresh,
        position_frac=C.POSITION_FRAC,
        use_kelly=C.USE_KELLY,
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

    # ── Save artifacts ────────────────────────────────
    trade_log = bt.get_trade_log()
    trade_log.to_csv(f"{out_dir}/trade_log.csv", index=False)

    eq = bt.get_equity_curve()
    pd.DataFrame({"equity": eq}).to_csv(f"{out_dir}/equity_curve.csv", index=False)

    pd.Series(selected_features).to_csv(f"{out_dir}/selected_features.csv", index=False)

    pd.DataFrame(history.history).to_csv(f"{out_dir}/training_history.csv", index=False)

    # Save run summary JSON
    save_run_summary(
        out_dir, report,
        args_info=date_info,
        model_params=total_params,
        epochs_run=epochs_run,
        best_val_loss=round(best_val_loss, 4),
        best_val_acc=round(best_val_acc, 4),
        n_features=len(selected_features),
        data_shape=df.shape,
        calibrated_long=long_thresh,
        calibrated_short=short_thresh,
    )

    # Generate charts
    generate_charts(out_dir, eq, trade_log, history.history)

    print(f"\n  Artifacts saved to {out_dir}/")
    print(f"  - run_summary.json")
    print(f"  - trade_log.csv ({len(trade_log)} rows, {len(trade_log.columns)} columns)")
    print(f"  - equity_curve.csv")
    print(f"  - selected_features.csv")
    print(f"  - training_history.csv")
    print(f"  Charts:")
    print(f"  - equity_curve.png")
    print(f"  - pnl_distribution.png")
    print(f"  - training_curves.png")
    print(f"  - cumulative_pnl.png")
    print(f"  - prediction_confidence.png")
    print(f"  - rolling_win_rate.png")
    print(f"  - action_breakdown.png")

    # Print last 10 trades
    print("\n  LAST 10 EVENTS:")
    print(f"  {'Timestamp':<22} {'Signal':<7} {'Event':<12} {'Dir':<6} {'Price':>10} {'PnL $':>9} {'PosID':>5} {'Equity':>10}")
    for _, row in trade_log.tail(10).iterrows():
        ts_str = str(row["timestamp"])[:19] if row["timestamp"] else "N/A"
        direction = row["direction"] if row["direction"] else "-"
        pnl = row["pnl_amount"] if row["pnl_amount"] else 0.0
        print(f"  {ts_str:<22} {row['action']:<7} {row['event_type']:<12} {direction:<6} "
              f"{row['price']:>10.2f} {pnl:>+9.2f} "
              f"{row['position_id']:>5} ${row['total_equity']:>9.2f}")

    return model, report, trade_log


if __name__ == "__main__":
    model, report, trade_log = main()
