#main.py
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

from data_pipeline import create_target, get_data, create_features, clean_data
from rule_based_strategy import RuleBasedStrategy
from logistic_regression_strategy import LogisticRegressionStrategy
from backtest import group_trades_by_interval, run_backtest, run_backtest_with_probabilities
from backtest_result import BacktestResult


if __name__ == "__main__":
    df = get_data()
    df = create_features(df)
    df = create_target(df)
    df = clean_data(df)

    # Split data chronologically
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    
    
    train_data = df.iloc[:train_end]
    val_data = df.iloc[train_end:val_end]
    test_data = df.iloc[val_end:]
    print(f"Storlek på test_data: {len(test_data)}")
    # Train on rule-based buy signals only
    strategy_rule_based = RuleBasedStrategy()
    rule_signals = train_data.apply(strategy_rule_based.generate_signal, axis=1)
    training_data = train_data[rule_signals == 1]

    strategy_logistic_regression = LogisticRegressionStrategy()
    
    best_C = strategy_logistic_regression.tune(training_data)
    strategy_logistic_regression.train(training_data, C=best_C)

    # Baseline backtest on test data
    trades_rule_based = run_backtest(strategy_rule_based, test_data)
    if trades_rule_based is not None:
        results_rule_based = BacktestResult(trades_rule_based, "RuleBasedStrategy")
        results_rule_based.print_results()
    else:
        print("RuleBasedStrategy: No trades executed during test period.")

    # Interval backtest on test data
    trades_with_probabilities = run_backtest_with_probabilities(strategy_rule_based, strategy_logistic_regression, test_data)
    if trades_with_probabilities is not None:
        interval_results = group_trades_by_interval(trades_with_probabilities)
        BacktestResult.print_interval_table(interval_results)
    else:
        print("No trades executed during interval backtest.")
  