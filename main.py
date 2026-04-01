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

from data_pipeline import get_data, create_features
from rule_based_strategy import RuleBasedStrategy
from logistic_regression_strategy import LogisticRegressionStrategy
from backtest import run_backtest

df = get_data()
#print(df[['time', 'close', 'RSI', 'SMA_50', 'SMA_200', 'OBV']].tail(10))
df = create_features(df)

strategy_rule_based = RuleBasedStrategy()
df = strategy_rule_based.create_labels(df)

strategy_logistic_regression = LogisticRegressionStrategy(model=None) # TODO: add model
strategy_logistic_regression.train(df)

results_rule_based = run_backtest(strategy_rule_based, df)
results_logistic_regression = run_backtest(strategy_logistic_regression, df)