# Runs everything

# Psudo code: 
# df = get_data()
# strategyA = RuleBasedStrategy()
# strategyB = LogisticRegressionStrategy()
# strategyB.train(df)
#
# resultsA = run_backtest(strategyA, df)
# resultsB = run_backtest(strategyB, df)

# Compare and print results

import sys

from data_pipeline import get_data

df = get_data()
print(df[['time', 'close', 'RSI', 'SMA_50', 'SMA_200', 'OBV']].tail(10))
