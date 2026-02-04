class IntradaySimulator:
    def __init__(self, initial_capital=10000, fee=0.001):
        self.capital = initial_capital
        self.fee = fee # 0.1% Binance fee

    def simulate_trade(self, prediction, current_price, next_hour_price):
        # 1: Long, 0: Hold, -1: Short
        if prediction == 1:
            profit = (next_hour_price - current_price) / current_price
            self.capital *= (1 + profit - self.fee)
        elif prediction == -1:
            profit = (current_price - next_hour_price) / current_price
            self.capital *= (1 + profit - self.fee)
        return self.capital