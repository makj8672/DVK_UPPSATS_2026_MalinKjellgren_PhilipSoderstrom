# Standalone rule-based trading strategy

# Psudo code: 
# FUNCTION generate_signal(row):
#     TODO

class RuleBasedStrategy:
    # Runs on an individual row in real time or during backtest.
    # Checks at the latest row and desides whether to buy, sell or hold based on the indicators
    def generate_signal(self, row):
        cond1 = row["price_to_sma200"] > 0                  # Price over SMA200
        cond2 = row["sma_cross"] > 0                        # SMA50 over SMA200
        cond3 = 35 <= row["rsi"] <= 65                            # RSI neutral
        cond4 = row["obv_diff"] > 0                        # OBV stiger      
        
        if cond1 and cond2 and cond3 and cond4:
            return 1    # Buy signal
        return 0        # Hold signal

    # Runs through the entire dataframe during the "training of the modell"
    def create_labels(self, data_frame):
        data_frame["obv_diff"] = data_frame["OBV"].diff() # Calculate and save
        
        cond1 = data_frame["price_to_sma200"] > 0                       # Price over SMA200
        cond2 = data_frame["sma_cross"] > 0                             # SMA50 over SMA200
        cond3 = (data_frame["rsi"] >= 35) & (data_frame["rsi"] <= 65)   # RSI neutral
        cond4 = data_frame["obv_diff"] > 0                              # OBV rising

        data_frame["target"] = (cond1 & cond2 & cond3 & cond4).astype(int)
        print(data_frame["target"].value_counts())
        return data_frame
    

