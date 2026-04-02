#backtest.py

# Compare both strategies 

# Psudo code: 
# FUNCTION run_backtest(strategy, data):
#     FOR each row in dataframe:
#         signal = strategy.generate_signal(row)
#         Execute trade based on signal
#         Track profit/loss
#
#     Calculate win rate, profit factor,
#     Sharpe ratio, max drawdown
#     Return results

stop_loss_pct = 1.0
take_profit_pct = 2.0
forward_hours = 24

def run_backtest(strategy, data_frame):
    test_data = data_frame.iloc[int(len(data_frame) * 0.8):]  # Use last 20% of data for testing
    trades = []

    for i in range(len(test_data) - forward_hours):
        row = test_data.iloc[i]
        signal = strategy.generate_signal(row) # Get signal from strategy

        if signal == 1:  # Buy signal
            entry_price = row["close"]
            exit_price = entry_price

            for h in range(1, forward_hours + 1):
                if i + h >= len(test_data):
                    break
                current_price = test_data.iloc[i + h]["close"]
                current_return = (current_price - entry_price) / entry_price * 100

                if current_return <= -stop_loss_pct:
                    exit_price = current_price
                    break
                if current_return >= take_profit_pct:
                    exit_price = current_price
                    break
                if h == forward_hours:
                    exit_price = current_price

            return_pct = (exit_price - entry_price) / entry_price * 100
            trades.append(return_pct)

    if len(trades) == 0:
        print("No trades done during test period.")
        return None
    
    return trades
    