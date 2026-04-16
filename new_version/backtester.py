"""
Backtester / Trading Simulator
- Multi-position management via TradeRouter
- LONG/SHORT signals open new positions; opposite signals close existing ones
- HOLD signals keep all positions open
- Mark-to-market equity curve with unrealized PnL
- Per-trade logging with position lifecycle (open/close times, duration)
- PnL, Sharpe, Max Drawdown computation
- Optional Kelly criterion position sizing
"""

import numpy as np
import pandas as pd
import config as C
from trade_router import TradeRouter


class Backtester:
    def __init__(
        self,
        initial_capital: float = C.INITIAL_CAPITAL,
        fee: float = C.TRADE_FEE,
        long_thresh: float = C.LONG_THRESHOLD,
        short_thresh: float = C.SHORT_THRESHOLD,
        position_frac: float = C.POSITION_FRAC,
        use_kelly: bool = C.USE_KELLY,
        symbol: str = C.SYMBOL,
    ):
        self.initial_capital = initial_capital
        self.fee = fee
        self.long_thresh = long_thresh
        self.short_thresh = short_thresh
        self.position_frac = position_frac
        self.use_kelly = use_kelly
        self.symbol = symbol

        # Trade router handles positions and capital
        self.router = TradeRouter(
            initial_capital=initial_capital,
            fee=fee,
            position_frac=position_frac,
            symbol=symbol,
        )

        self.trade_log = []
        self.equity_curve = [initial_capital]
        self._peak_equity = initial_capital
        self._closed_trades = 0
        self._closed_wins = 0
        self._cumulative_pnl = 0.0

    def _kelly_fraction(self, prob: float) -> float:
        """Half-Kelly sizing based on model confidence, capped at position_frac."""
        confidence = abs(prob - 0.5) * 2
        kelly = 0.5 * confidence
        return min(kelly, self.position_frac)

    def step(self, prob: float, price: float, timestamp=None,
             actual_direction: int = None):
        """
        Execute one trading step with position management.

        prob: model's P(price goes up)
        price: current close price
        timestamp: current timestamp
        actual_direction: actual price movement (1=UP, 0=DOWN)
        """
        # Determine signal from probability
        if prob > self.long_thresh:
            signal = "LONG"
        elif prob < self.short_thresh:
            signal = "SHORT"
        else:
            signal = "HOLD"

        # Optionally adjust position fraction via Kelly
        if self.use_kelly:
            self.router.position_frac = self._kelly_fraction(prob)
        else:
            self.router.position_frac = self.position_frac

        # Process signal through router
        events = self.router.process_signal(signal, price, timestamp)

        # Directional prediction
        pred_dir = 1 if prob > 0.5 else 0
        correct = (pred_dir == actual_direction) if actual_direction is not None else None

        # Mark-to-market equity
        equity = self.router.get_total_equity(price)
        self.equity_curve.append(equity)

        # Peak tracking for drawdown
        if equity > self._peak_equity:
            self._peak_equity = equity
        current_dd = (equity - self._peak_equity) / (self._peak_equity + 1e-10)

        # Portfolio return
        portfolio_return_pct = (equity - self.initial_capital) / self.initial_capital * 100

        # Process events and create log entries
        if len(events) == 0:
            # HOLD signal with no events — log it
            self._log_hold(signal, prob, price, pred_dir, actual_direction,
                           correct, equity, current_dd, portfolio_return_pct, timestamp)
        else:
            for event in events:
                self._log_event(event, signal, prob, price, pred_dir,
                                actual_direction, correct, equity, current_dd,
                                portfolio_return_pct, timestamp)

    def _log_hold(self, signal, prob, price, pred_dir, actual_direction,
                  correct, equity, current_dd, portfolio_return_pct, timestamp):
        """Log a HOLD step (no position changes)."""
        self.trade_log.append({
            "symbol": self.symbol,
            "timestamp": timestamp,
            "action": signal,
            "prob": round(prob, 6),
            "price": round(price, 6),
            "pred_direction": pred_dir,
            "actual_direction": actual_direction,
            "correct": correct,
            # Position lifecycle
            "position_id": 0,
            "event_type": "HOLD",
            "direction": None,
            "entry_price": round(price, 6),
            "close_price": None,
            "position_open_time": None,
            "position_close_time": None,
            "position_duration_hours": None,
            # Financials
            "allocated_capital": 0.0,
            "pnl_pct": 0.0,
            "pnl_amount": 0.0,
            "fee_paid": 0.0,
            # Portfolio state
            "open_positions_count": self.router.num_open_positions,
            "available_capital": round(self.router.available_capital, 2),
            "total_equity": round(equity, 2),
            "peak_equity": round(self._peak_equity, 2),
            "drawdown_pct": round(current_dd * 100, 2),
            "cumulative_pnl": round(self._cumulative_pnl, 2),
            "cumulative_closed_trades": self._closed_trades,
            "portfolio_return_pct": round(portfolio_return_pct, 2),
        })

    def _log_event(self, event, signal, prob, price, pred_dir,
                   actual_direction, correct, equity, current_dd,
                   portfolio_return_pct, timestamp):
        """Log a trade event (OPEN, CLOSE, FORCE_CLOSE)."""
        is_close = event["event_type"] in ("CLOSE", "FORCE_CLOSE")

        if is_close:
            self._closed_trades += 1
            self._cumulative_pnl += event["pnl_amount"]
            # Determine if this close was profitable (direction-correct)
            if event["direction"] == "LONG":
                position_correct = (event["close_price"] > event["entry_price"])
            else:
                position_correct = (event["close_price"] < event["entry_price"])
            if position_correct:
                self._closed_wins += 1

        self.trade_log.append({
            "symbol": self.symbol,
            "timestamp": timestamp,
            "action": signal,
            "prob": round(prob, 6),
            "price": round(price, 6),
            "pred_direction": pred_dir,
            "actual_direction": actual_direction,
            "correct": correct,
            # Position lifecycle
            "position_id": event["position_id"],
            "event_type": event["event_type"],
            "direction": event["direction"],
            "entry_price": round(event["entry_price"], 6),
            "close_price": round(event["close_price"], 6) if event["close_price"] is not None else None,
            "position_open_time": event["position_open_time"],
            "position_close_time": event["position_close_time"],
            "position_duration_hours": round(event["position_duration_hours"], 1) if event["position_duration_hours"] is not None else None,
            # Financials
            "allocated_capital": round(event["allocated_capital"], 2),
            "pnl_pct": round(event["pnl_pct"], 4) if is_close else 0.0,
            "pnl_amount": round(event["pnl_amount"], 2) if is_close else 0.0,
            "fee_paid": round(event.get("exit_fee", 0.0), 4),
            # Portfolio state
            "open_positions_count": self.router.num_open_positions,
            "available_capital": round(self.router.available_capital, 2),
            "total_equity": round(equity, 2),
            "peak_equity": round(self._peak_equity, 2),
            "drawdown_pct": round(current_dd * 100, 2),
            "cumulative_pnl": round(self._cumulative_pnl, 2),
            "cumulative_closed_trades": self._closed_trades,
            "portfolio_return_pct": round(portfolio_return_pct, 2),
        })

    def run(self, probs: np.ndarray, prices: np.ndarray,
            timestamps=None, actuals: np.ndarray = None):
        """
        Run full backtest over arrays.
        probs: (N,) model probabilities
        prices: (N+1,) close prices (need one extra for force-close at end)
        """
        n = len(probs)
        for i in range(n):
            ts = timestamps[i] if timestamps is not None else i
            actual = int(actuals[i]) if actuals is not None else None
            self.step(probs[i], prices[i], ts, actual)

        # Force-close any remaining open positions at the last price
        if self.router.num_open_positions > 0:
            last_price = prices[n] if n < len(prices) else prices[-1]
            last_ts = timestamps[-1] if timestamps is not None else n
            force_events = self.router.force_close_all(last_price, last_ts)
            equity = self.router.get_total_equity(last_price)
            for event in force_events:
                self._closed_trades += 1
                self._cumulative_pnl += event["pnl_amount"]
                self.trade_log.append({
                    "symbol": self.symbol,
                    "timestamp": last_ts,
                    "action": "FORCE_CLOSE",
                    "prob": 0.0,
                    "price": round(last_price, 6),
                    "pred_direction": None,
                    "actual_direction": None,
                    "correct": None,
                    "position_id": event["position_id"],
                    "event_type": "FORCE_CLOSE",
                    "direction": event["direction"],
                    "entry_price": round(event["entry_price"], 6),
                    "close_price": round(event["close_price"], 6),
                    "position_open_time": event["position_open_time"],
                    "position_close_time": event["position_close_time"],
                    "position_duration_hours": round(event["position_duration_hours"], 1) if event["position_duration_hours"] is not None else None,
                    "allocated_capital": round(event["allocated_capital"], 2),
                    "pnl_pct": round(event["pnl_pct"], 4),
                    "pnl_amount": round(event["pnl_amount"], 2),
                    "fee_paid": round(event.get("exit_fee", 0.0), 4),
                    "open_positions_count": 0,
                    "available_capital": round(self.router.available_capital, 2),
                    "total_equity": round(equity, 2),
                    "peak_equity": round(self._peak_equity, 2),
                    "drawdown_pct": 0.0,
                    "cumulative_pnl": round(self._cumulative_pnl, 2),
                    "cumulative_closed_trades": self._closed_trades,
                    "portfolio_return_pct": round((equity - self.initial_capital) / self.initial_capital * 100, 2),
                })
            self.equity_curve.append(equity)

        return self.get_report()

    def get_report(self) -> dict:
        log_df = pd.DataFrame(self.trade_log)
        eq = np.array(self.equity_curve)

        # Final equity is mark-to-market (all positions should be closed by now)
        final_equity = eq[-1] if len(eq) > 0 else self.initial_capital

        # Closed trades for PnL analysis
        closed = log_df[log_df["event_type"].isin(["CLOSE", "FORCE_CLOSE"])]

        # Directional accuracy on closed trades
        if len(closed) > 0:
            trade_pnls = closed["pnl_pct"]
            dir_acc = (trade_pnls > 0).mean()
        else:
            dir_acc = 0.0

        # All predictions accuracy (every step)
        all_with_actual = log_df[log_df["actual_direction"].notna()]
        total_dir_acc = all_with_actual["correct"].mean() if len(all_with_actual) > 0 else 0.0

        # PnL
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # Sharpe (hourly returns annualized)
        returns = np.diff(eq) / (eq[:-1] + 1e-10)
        returns = returns[np.isfinite(returns)]
        sharpe = (returns.mean() / (returns.std() + 1e-10)) * np.sqrt(8760)

        # Max Drawdown
        running_max = np.maximum.accumulate(eq)
        drawdowns = (eq - running_max) / (running_max + 1e-10)
        max_dd = drawdowns.min()

        n_closed = len(closed)
        n_longs = len(closed[closed["direction"] == "LONG"])
        n_shorts = len(closed[closed["direction"] == "SHORT"])

        # Count OPEN events and HOLD events
        n_opens = len(log_df[log_df["event_type"] == "OPEN"])
        n_holds = len(log_df[log_df["event_type"] == "HOLD"])

        max_equity = float(eq.max())

        if n_closed > 0:
            traded_pnl = closed["pnl_pct"]
            biggest_win = float(traded_pnl.max())
            biggest_loss = float(traded_pnl.min())
            avg_trade_pnl = float(traded_pnl.mean())
            biggest_trade_amount = float(closed["allocated_capital"].max())
            total_fees = float(log_df["fee_paid"].sum())

            winning = traded_pnl[traded_pnl > 0]
            losing = traded_pnl[traded_pnl < 0]
            win_rate = len(winning) / n_closed
            gross_profit = winning.sum() if len(winning) > 0 else 0.0
            gross_loss = abs(losing.sum()) if len(losing) > 0 else 0.0
            profit_factor = float(gross_profit / (gross_loss + 1e-10))

            # Streak tracking on closed trades
            streaks = (traded_pnl > 0).astype(int).values
            max_win_streak = max_loss_streak = cur_win = cur_loss = 0
            for s in streaks:
                if s == 1:
                    cur_win += 1; cur_loss = 0
                    max_win_streak = max(max_win_streak, cur_win)
                else:
                    cur_loss += 1; cur_win = 0
                    max_loss_streak = max(max_loss_streak, cur_loss)

            # Average position duration
            durations = closed["position_duration_hours"].dropna()
            avg_duration = float(durations.mean()) if len(durations) > 0 else 0.0
        else:
            biggest_win = 0.0
            biggest_loss = 0.0
            avg_trade_pnl = 0.0
            biggest_trade_amount = 0.0
            total_fees = 0.0
            win_rate = 0.0
            profit_factor = 0.0
            max_win_streak = 0
            max_loss_streak = 0
            avg_duration = 0.0

        report = {
            "initial_capital": self.initial_capital,
            "final_capital": round(final_equity, 2),
            "total_return_pct": round(total_return * 100, 2),
            "max_equity": round(max_equity, 2),
            "directional_accuracy_trades": round(dir_acc * 100, 2),
            "directional_accuracy_all": round(total_dir_acc * 100, 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "total_trades": n_closed,
            "n_longs": n_longs,
            "n_shorts": n_shorts,
            "n_holds": n_holds,
            "n_positions_opened": n_opens,
            "win_rate_pct": round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 4),
            "avg_trade_pnl_pct": round(avg_trade_pnl, 4),
            "biggest_win_pct": round(biggest_win, 4),
            "biggest_loss_pct": round(biggest_loss, 4),
            "biggest_trade_amount": round(biggest_trade_amount, 2),
            "total_fees_paid": round(total_fees, 2),
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "avg_position_duration_hours": round(avg_duration, 1),
        }
        return report

    def get_trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_log)

    def get_equity_curve(self) -> np.ndarray:
        return np.array(self.equity_curve)
