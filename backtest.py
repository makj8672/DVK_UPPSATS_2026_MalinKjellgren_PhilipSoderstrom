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
