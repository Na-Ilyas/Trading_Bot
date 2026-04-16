import logging

class IntradaySimulator:
    def __init__(self, initial_balance=1000):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.fee = 0.001       # 0.1% Binance taker fee
        self.slippage = 0.0005 # 0.05% slippage penalty
        self.trade_count = 0

    def calculate_position_size(self, prob, threshold, max_risk_pct=0.10, min_notional=10.0):
        """Dynamic sizing linear scaling based on confidence."""
        if prob <= threshold:
            return 0.0
            
        confidence_scalar = (prob - threshold) / (1.0 - threshold)
        position_size = self.balance * max_risk_pct * confidence_scalar
        
        if position_size < min_notional:
            return 0.0
        return position_size

    def validate_prediction(self, model_prob, current_price, next_hour_price, 
                            long_threshold, short_threshold):
        trade_executed = False
        pnl = 0
        action = "HOLD"
        size = 0

        # LONG Logic
        if model_prob > long_threshold:
            size = self.calculate_position_size(model_prob, long_threshold)
            if size > 0:
                # Calculate return with slippage on entry and exit
                change = (next_hour_price - (current_price * (1 + self.slippage))) / current_price
                gross_pnl = size * change
                fee_cost = size * self.fee * 2 # Entry and Exit fee
                pnl = gross_pnl - fee_cost
                
                self.balance += pnl
                action = "LONG"
                trade_executed = True
            
        # SHORT Logic
        elif model_prob < short_threshold:
            # 1.0 - prob reflects the confidence in the SHORT
            size = self.calculate_position_size((1.0 - model_prob), (1.0 - short_threshold))
            if size > 0:
                change = ((current_price * (1 - self.slippage)) - next_hour_price) / current_price
                gross_pnl = size * change
                fee_cost = size * self.fee * 2
                pnl = gross_pnl - fee_cost
                
                self.balance += pnl
                action = "SHORT"
                trade_executed = True

        if trade_executed:
            self.trade_count += 1
            logging.info(f"[{action}] Size: ${size:.2f} | Prob: {model_prob:.3f} | PnL: ${pnl:.2f} | Bal: ${self.balance:.2f}")

        return self.balance