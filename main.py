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

from data_pipeline import create_target, get_data, create_features, clean_data, split_data
from rule_based_strategy import RuleBasedStrategy
from logistic_regression_strategy import LogisticRegressionStrategy
from backtest import group_trades_by_interval, run_backtest, run_backtest_with_probabilities
from backtest_result import BacktestResult


if __name__ == "__main__":
    df = get_data()
    df = create_features(df)
    df = create_target(df)
    df = clean_data(df)

    train_data, val_data, test_data = split_data(df)
    print(f"Storlek på test_data: {len(test_data)}")
    
    strategy_rule_based = RuleBasedStrategy()
    strategy_logistic_regression = LogisticRegressionStrategy()

    # Filtering buy signals for training
    rule_signals = train_data.apply(strategy_rule_based.generate_signal, axis=1)
    buy_signal_rows = train_data[rule_signals == 1]

    val_signals = val_data.apply(strategy_rule_based.generate_signal, axis=1)
    buy_signal_rows_val = val_data[val_signals == 1]
    
    # Tune C parameter on validation data and train logistic regression model
    best_C = strategy_logistic_regression.tune(buy_signal_rows, buy_signal_rows_val)
    strategy_logistic_regression.train(buy_signal_rows, buy_signal_rows_val, C=best_C)

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

    # Backtest logistic regression strategy on test data
    trades_lr = run_backtest(strategy_logistic_regression, test_data)
    if trades_lr is not None:
        results_lr = BacktestResult(trades_lr, "LogisticRegressionStrategy")
        results_lr.print_results()
    else:
        print("LogisticRegressionStrategy: No trades executed.")
    