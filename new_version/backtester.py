"""
Backtester / Trading Simulator
- Adjustable position fraction (risk per trade)
- Long / Short / Hold logic based on model probability
- Per-trade logging with directional accuracy
- PnL, Sharpe, Max Drawdown computation
"""

import numpy as np
import pandas as pd
import config as C


class Backtester:
    def __init__(
        self,
        initial_capital: float = C.INITIAL_CAPITAL,
        fee: float = C.TRADE_FEE,
        long_thresh: float = C.LONG_THRESHOLD,
        short_thresh: float = C.SHORT_THRESHOLD,
        position_frac: float = C.POSITION_FRAC,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.fee = fee
        self.long_thresh = long_thresh
        self.short_thresh = short_thresh
        self.position_frac = position_frac  # fraction of capital per trade

        self.trade_log = []
        self.equity_curve = [initial_capital]

    def step(self, prob: float, entry_price: float, exit_price: float,
             timestamp=None, actual_direction: int = None):
        """
        Execute one trading step.
        prob: model's P(price goes up)
        entry_price: price at current hour
        exit_price: price at next hour (for simulation)
        """
        action = "HOLD"
        pnl_pct = 0.0
        trade_capital = self.capital * self.position_frac

        if prob > self.long_thresh:
            # LONG: profit if price goes up
            action = "LONG"
            pnl_pct = (exit_price - entry_price) / entry_price - self.fee
        elif prob < self.short_thresh:
            # SHORT: profit if price goes down
            action = "SHORT"
            pnl_pct = (entry_price - exit_price) / entry_price - self.fee

        # Update capital
        if action != "HOLD":
            trade_pnl = trade_capital * pnl_pct
            self.capital += trade_pnl
            # Ensure capital doesn't go negative (liquidation guard)
            self.capital = max(self.capital, 0.0)

        self.equity_curve.append(self.capital)

        # Determine prediction correctness
        pred_dir = 1 if prob > 0.5 else 0
        correct = (pred_dir == actual_direction) if actual_direction is not None else None

        self.trade_log.append({
            "timestamp": timestamp,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "prob": prob,
            "action": action,
            "pred_direction": pred_dir,
            "actual_direction": actual_direction,
            "correct": correct,
            "pnl_pct": pnl_pct if action != "HOLD" else 0.0,
            "capital": self.capital,
        })

        return self.capital

    def run(self, probs: np.ndarray, prices: np.ndarray,
            timestamps=None, actuals: np.ndarray = None):
        """
        Run full backtest over arrays.
        probs: (N,) model probabilities
        prices: (N+1,) close prices (need one extra for exit of last trade)
        """
        n = len(probs)
        for i in range(n):
            ts = timestamps[i] if timestamps is not None else i
            actual = int(actuals[i]) if actuals is not None else None
            self.step(probs[i], prices[i], prices[i + 1], ts, actual)
        return self.get_report()

    def get_report(self) -> dict:
        log_df = pd.DataFrame(self.trade_log)
        eq = np.array(self.equity_curve)

        # Directional accuracy (only on non-HOLD trades with known actuals)
        traded = log_df[log_df["action"] != "HOLD"]
        if len(traded) > 0 and traded["correct"].notna().any():
            dir_acc = traded["correct"].mean()
        else:
            dir_acc = 0.0

        # All predictions accuracy (including HOLD)
        all_with_actual = log_df[log_df["actual_direction"].notna()]
        total_dir_acc = all_with_actual["correct"].mean() if len(all_with_actual) > 0 else 0.0

        # PnL
        total_return = (self.capital - self.initial_capital) / self.initial_capital

        # Sharpe (hourly returns annualized)
        returns = np.diff(eq) / eq[:-1]
        returns = returns[np.isfinite(returns)]
        sharpe = (returns.mean() / (returns.std() + 1e-10)) * np.sqrt(8760)  # hourly → annual

        # Max Drawdown
        running_max = np.maximum.accumulate(eq)
        drawdowns = (eq - running_max) / (running_max + 1e-10)
        max_dd = drawdowns.min()

        n_trades = len(traded)
        n_longs  = len(traded[traded["action"] == "LONG"])
        n_shorts = len(traded[traded["action"] == "SHORT"])

        report = {
            "initial_capital": self.initial_capital,
            "final_capital": round(self.capital, 2),
            "total_return_pct": round(total_return * 100, 2),
            "directional_accuracy_trades": round(dir_acc * 100, 2),
            "directional_accuracy_all": round(total_dir_acc * 100, 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "total_trades": n_trades,
            "n_longs": n_longs,
            "n_shorts": n_shorts,
            "n_holds": len(log_df) - n_trades,
        }
        return report

    def get_trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_log)

    def get_equity_curve(self) -> np.ndarray:
        return np.array(self.equity_curve)
