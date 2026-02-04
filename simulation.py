class IntradaySimulator:
    def __init__(self, initial_balance=1000):
        self.balance = initial_balance
        self.fee = 0.001 # 0.1% Binance fee

    def validate_prediction(self, model_prob, current_price, next_hour_price, 
                            long_threshold=0.7, short_threshold=0.3):
        """
        model_prob: Raw probability from sigmoid (0 to 1)
        """
        # LONG Logic
        if model_prob > long_threshold:
            change = (next_hour_price - current_price) / current_price
            # Profit = Balance * (1 + %Change - Fee)
            self.balance *= (1 + change - self.fee)
            
        # SHORT Logic
        elif model_prob < short_threshold:
            change = (current_price - next_hour_price) / current_price
            # Shorting: You profit if price goes DOWN
            self.balance *= (1 + change - self.fee)
            
        # HOLD Logic (Between 0.3 and 0.7) -> No change to balance
            
        return self.balance