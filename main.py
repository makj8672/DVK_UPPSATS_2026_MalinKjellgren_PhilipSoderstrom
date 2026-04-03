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
from backtest import run_backtest, run_backtest_all_intervals
from backtest_result import BacktestResult


if __name__ == "__main__":
    df = get_data()
    #print(df[['time', 'close', 'RSI', 'SMA_50', 'SMA_200', 'OBV']].tail(10))
    df = create_features(df)
    df = create_target(df)
    df = clean_data(df)
  
    
    

    strategy_rule_based = RuleBasedStrategy()
    # df = strategy_rule_based.create_labels(df)

    strategy_logistic_regression = LogisticRegressionStrategy(model=None) # TODO: add model
    
    rule_signals = df.apply(strategy_rule_based.generate_signal, axis=1)
    training_data = df[rule_signals == 1]  # Use only rows where rule-based strategy says buy for training
    
    strategy_logistic_regression.train(training_data)

    #TODO Jag la in ännu ett vis att kör backtest på, vi får bestämma vilken  eller kombo 
    # Backtest with calling function from other file
    #results_rule_based = run_backtest(strategy_rule_based, df)
    #results_logistic_regression = run_backtest(strategy_logistic_regression, df)

    # Backtest with calling function from other file and saving results in BacktestResult class
    #results_rule_based = BacktestResult(run_backtest(strategy_rule_based, df), "RuleBasedStrategy")
    #results_logistic_regression = BacktestResult(run_backtest(strategy_logistic_regression, df), "LogisticRegressionStrategy")

    trades_rule_based = run_backtest(strategy_rule_based, df)
    trades_logistic_regression = run_backtest(strategy_logistic_regression, df)

    if trades_rule_based is not None:
        results_rule_based = BacktestResult(trades_rule_based, "RuleBasedStrategy")
        results_rule_based.print_results()
    else:
        print("RuleBasedStrategy: No trades executed during test period.")
    
    # Backtesting LogistcRegressionStrategy per interval
    interval_results = run_backtest_all_intervals(strategy_rule_based, strategy_logistic_regression, df)
    BacktestResult.print_interval_table(interval_results)
