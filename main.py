#main.py
# Runs everything

from data_pipeline import create_target, get_data, create_features, clean_data, split_data
from rule_based_strategy import RuleBasedStrategy
from logistic_regression_strategy import LogisticRegressionStrategy
from backtest import group_trades_by_interval, run_backtest, run_backtest_with_probabilities
from backtest_result import BacktestResult

import numpy as np


def _build_rule_trading_split(frame, rule_signals):
    """Filter out rows where rule == 0 and tells LR which 
    direction we wish to predict (long vs short) via signal_sign."""
    
    """Rows where rule != 0; target = next-bar success in that direction; add signal_sign."""
    mask = rule_signals != 0        # Only keep rows where rule fires (long or short)
    out = frame.loc[mask].copy()    # Copy to avoid SettingWithCopyWarning when adding columns
    sig = rule_signals.loc[mask]    # Corresponding signals for the filtered rows

    # Long: next close up. Short: next close down (strict inequality matches create_target). #TODO: Kolla på denna och förstå
    out["target"] = np.where( #TODO: Kanske bättre att ha två separata target-kolumner, en för long och en för short, istället för att blanda i en gemensam target-kolumn? Då kan vi undvika att behöva använda signal_sign som input till LR och istället bara träna på rätt target-kolumn beroende på signal
        sig.to_numpy() == 1,                
        out["target"].to_numpy(),
        out["target_next_down"].to_numpy(),
    )
    out["signal_sign"] = np.where(sig.to_numpy() == 1, 1.0, -1.0)
    return out, sig


if __name__ == "__main__":
    df = get_data()
    df = create_features(df)
    df = create_target(df)
    df = clean_data(df)

    train_data, val_data, test_data = split_data(df)
    print(f"Storlek på test_data: {len(test_data)}")

    strategy_rule_based = RuleBasedStrategy()
    strategy_logistic_regression = LogisticRegressionStrategy()

    rule_train = train_data.apply(strategy_rule_based.generate_signal, axis=1)
    training_data, _ = _build_rule_trading_split(train_data, rule_train)
    print("\nTräningssignaler (1=köp, -1=sälj):")
    print(rule_train[rule_train != 0].value_counts())

    rule_val = val_data.apply(strategy_rule_based.generate_signal, axis=1)
    val_trading_data, _ = _build_rule_trading_split(val_data, rule_val)

    best_C = strategy_logistic_regression.tune(training_data, val_trading_data)
    strategy_logistic_regression.train(training_data, val_trading_data, C=best_C)

    trades_rule_based = run_backtest(strategy_rule_based, test_data)
    if trades_rule_based is not None:
        results_rule_based = BacktestResult(trades_rule_based, "RuleBasedStrategy")
        results_rule_based.print_results()
    else:
        print("RuleBasedStrategy: No trades executed during test period.")

    trades_with_probabilities = run_backtest_with_probabilities(
        strategy_rule_based, strategy_logistic_regression, test_data
    )
    if trades_with_probabilities is not None:
        interval_results = group_trades_by_interval(trades_with_probabilities)
        BacktestResult.print_interval_table(interval_results)
    else:
        print("No trades executed during interval backtest.")

    trades_lr = run_backtest(strategy_logistic_regression, test_data)
    if trades_lr is not None:
        results_lr = BacktestResult(trades_lr, "LogisticRegressionStrategy (rule + LR filter)")
        results_lr.print_results()
    else:
        print("LogisticRegressionStrategy: No trades executed.")
