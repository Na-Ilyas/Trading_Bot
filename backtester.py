class IntradaySimulator:
    def __init__(self, initial_capital=1000):
        self.balance = initial_capital
        self.fee = 0.001 # 0.1% Binance maker/taker fee

    def execute_simulation_hour(self, prediction_signal, entry_price, exit_price):
        """
        Simulates a Long/Short/Hold decision for exactly one hour.
        """
        # Threshold-based execution (ML Engineer perspective)
        if prediction_signal > 0.75: # Strong Bullish
            pnl = (exit_price - entry_price) / entry_price
            self.balance *= (1 + pnl - self.fee)
        elif prediction_signal < 0.25: # Strong Bearish
            pnl = (entry_price - exit_price) / entry_price
            self.balance *= (1 + pnl - self.fee)
        
        return self.balance