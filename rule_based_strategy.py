#rule_based_strategy.py
# Standalone rule-based trading strategy

class RuleBasedStrategy:
    """Simple rule-based strategy using technical indicators.
    Runs on an individual row in real time or during backtest.
    Checks at the latest row and desides whether to buy, sell or hold based on the indicators"""

    def generate_signal(self, row):
        cond1 = row["price_to_sma"] > 0     # Price over SMA200
        cond2 = row["sma_cross"] > 0        # SMA50 over SMA200
        cond3 = 35 <= row["rsi"] <= 65      # RSI neutral
        cond4 = row["obv_diff"] > 0         # OBV rising      
        
        if cond1 and cond2 and cond3 and cond4:
            return 1    # Buy signal
        return 0        # Hold signal

    def get_probability(self, row):
        return None

 

