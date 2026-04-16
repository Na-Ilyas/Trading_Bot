"""
Trade Router / Position Manager
════════════════════════════════
Manages multiple simultaneous positions with proper lifecycle tracking.

Rules:
  - LONG signal  → close ALL open SHORTs, then open a new LONG
  - SHORT signal → close ALL open LONGs, then open a new SHORT
  - HOLD signal  → do nothing (all positions stay open)
  - Positions are only closed by an OPPOSITE signal (or force-close at backtest end)
  - Multiple same-direction positions can coexist

Each position tracks: id, direction, entry price, entry time, allocated capital.
PnL is realized only on close. Equity curve uses mark-to-market for open positions.
"""

import numpy as np
import config as C


class TradeRouter:
    def __init__(
        self,
        initial_capital: float = C.INITIAL_CAPITAL,
        fee: float = C.TRADE_FEE,
        position_frac: float = C.POSITION_FRAC,
        symbol: str = C.SYMBOL,
    ):
        self.available_capital = initial_capital
        self.initial_capital = initial_capital
        self.fee = fee
        self.position_frac = position_frac
        self.symbol = symbol

        self.open_positions = []       # list of position dicts
        self._position_counter = 0
        self._realized_pnl = 0.0
        self._total_fees = 0.0

    # ── Position Lifecycle ────────────────────────────

    def _open_position(self, direction, price, timestamp):
        """Open a new position, allocating capital from the available pool."""
        allocated = self.available_capital * self.position_frac
        if allocated <= 0:
            return None  # No capital to allocate

        # Charge entry fee
        entry_fee = allocated * self.fee
        allocated_after_fee = allocated - entry_fee
        self.available_capital -= allocated
        self._total_fees += entry_fee

        self._position_counter += 1
        pos = {
            "id": self._position_counter,
            "direction": direction,
            "entry_price": price,
            "entry_time": timestamp,
            "allocated_capital": allocated,           # original allocation (before entry fee)
            "allocated_after_fee": allocated_after_fee,  # actual capital in position
        }
        self.open_positions.append(pos)
        return pos

    def _close_position(self, pos, price, timestamp):
        """Close a single position, realize PnL, return capital to pool."""
        if pos["direction"] == "LONG":
            pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]
        else:  # SHORT
            pnl_pct = (pos["entry_price"] - price) / pos["entry_price"]

        # PnL on the capital actually in the position
        pnl_amount = pos["allocated_after_fee"] * pnl_pct

        # Charge exit fee on the current value
        current_value = pos["allocated_after_fee"] + pnl_amount
        exit_fee = abs(current_value) * self.fee
        net_return = current_value - exit_fee

        self._realized_pnl += (net_return - pos["allocated_capital"])
        self._total_fees += exit_fee

        # Return capital to available pool
        self.available_capital += max(net_return, 0.0)

        # Compute duration
        duration_hours = None
        if pos["entry_time"] is not None and timestamp is not None:
            try:
                delta = timestamp - pos["entry_time"]
                duration_hours = delta.total_seconds() / 3600
            except (TypeError, AttributeError):
                duration_hours = None

        return {
            "position_id": pos["id"],
            "event_type": "CLOSE",
            "direction": pos["direction"],
            "entry_price": pos["entry_price"],
            "close_price": price,
            "position_open_time": pos["entry_time"],
            "position_close_time": timestamp,
            "position_duration_hours": duration_hours,
            "allocated_capital": pos["allocated_capital"],
            "pnl_pct": pnl_pct * 100,
            "pnl_amount": net_return - pos["allocated_capital"],
            "exit_fee": exit_fee,
            "capital_returned": max(net_return, 0.0),
        }

    # ── Signal Processing ─────────────────────────────

    def process_signal(self, signal, price, timestamp):
        """
        Process a trading signal. Returns a list of event dicts.

        signal: "LONG", "SHORT", or "HOLD"
        price: current close price
        timestamp: current timestamp
        """
        events = []

        if signal == "LONG":
            # Close all open SHORTs
            shorts = [p for p in self.open_positions if p["direction"] == "SHORT"]
            for pos in shorts:
                event = self._close_position(pos, price, timestamp)
                events.append(event)
            self.open_positions = [p for p in self.open_positions if p["direction"] != "SHORT"]

            # Open new LONG
            pos = self._open_position("LONG", price, timestamp)
            if pos is not None:
                events.append({
                    "position_id": pos["id"],
                    "event_type": "OPEN",
                    "direction": "LONG",
                    "entry_price": price,
                    "close_price": None,
                    "position_open_time": timestamp,
                    "position_close_time": None,
                    "position_duration_hours": None,
                    "allocated_capital": pos["allocated_capital"],
                    "pnl_pct": 0.0,
                    "pnl_amount": 0.0,
                    "exit_fee": 0.0,
                    "capital_returned": 0.0,
                })

        elif signal == "SHORT":
            # Close all open LONGs
            longs = [p for p in self.open_positions if p["direction"] == "LONG"]
            for pos in longs:
                event = self._close_position(pos, price, timestamp)
                events.append(event)
            self.open_positions = [p for p in self.open_positions if p["direction"] != "LONG"]

            # Open new SHORT
            pos = self._open_position("SHORT", price, timestamp)
            if pos is not None:
                events.append({
                    "position_id": pos["id"],
                    "event_type": "OPEN",
                    "direction": "SHORT",
                    "entry_price": price,
                    "close_price": None,
                    "position_open_time": timestamp,
                    "position_close_time": None,
                    "position_duration_hours": None,
                    "allocated_capital": pos["allocated_capital"],
                    "pnl_pct": 0.0,
                    "pnl_amount": 0.0,
                    "exit_fee": 0.0,
                    "capital_returned": 0.0,
                })

        # HOLD: do nothing — all positions stay open

        return events

    # ── Equity & Reporting ────────────────────────────

    def get_total_equity(self, current_price):
        """Mark-to-market: available capital + unrealized PnL of all open positions."""
        equity = self.available_capital
        for pos in self.open_positions:
            if pos["direction"] == "LONG":
                unrealized_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
            else:
                unrealized_pct = (pos["entry_price"] - current_price) / pos["entry_price"]
            equity += pos["allocated_after_fee"] * (1 + unrealized_pct)
        return equity

    def force_close_all(self, price, timestamp):
        """Force-close all open positions at the given price. Used at backtest end."""
        events = []
        for pos in self.open_positions:
            event = self._close_position(pos, price, timestamp)
            event["event_type"] = "FORCE_CLOSE"
            events.append(event)
        self.open_positions = []
        return events

    @property
    def num_open_positions(self):
        return len(self.open_positions)

    @property
    def total_fees(self):
        return self._total_fees

    @property
    def realized_pnl(self):
        return self._realized_pnl
