#backtest.py
# Compare both strategies across probability intervals

stop_loss_pct = 1.0
take_profit_pct = 2.0
forward_hours = 24

INTERVALS = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0)
]

def _execute_trade(test_data, i, entry_price, signal): #TODO: new - add signal parameter to differentiate between long and short trades, and adjust stop loss and take profit accordingly
    """Execute a single long trade (signal==1) or short trade (signal==-1) and return the return percentage."""
    exit_price = entry_price

    for h in range(1, forward_hours + 1):
        if i + h >= len(test_data):
            break
        current_price = test_data.iloc[i + h]["close"]

        #TODO: new - adjust return calculation for short trades
        if signal == 1:  # Long trade
            current_return = (current_price - entry_price) / entry_price * 100
        else:  # Short trade
            current_return = (entry_price - current_price) / entry_price * 100

        if current_return <= -stop_loss_pct:
            exit_price = current_price
            break
        if current_return >= take_profit_pct:
            exit_price = current_price
            break
        if h == forward_hours:
            exit_price = current_price
    
    # TODO: new - adjust return calculation for short trades
    if signal == 1:  # Long trade return
        return (exit_price - entry_price) / entry_price * 100
    return (exit_price - entry_price) / entry_price * 100


def run_backtest(strategy, data_frame):
    """Backtest for rule-based strategy as baseline.
    Expects pre-split test data."""
    trades = []

    for i in range(len(data_frame) - forward_hours):
        row = data_frame.iloc[i]
        signal = strategy.generate_signal(row)

        if signal == 1 or signal == -1: # TODO: new - include short trades
            entry_price = row["close"]
            return_pct = _execute_trade(data_frame, i, entry_price, signal) # TODO: new - pass signal to execute_trade to differentiate between long and short trades
            trades.append(return_pct)

    return trades if len(trades) > 0 else None


def run_backtest_with_probabilities(rule_strategy, lr_strategy, data_frame):
    """Run backtest on pre-split test data."""
    trades = []

    for i in range(len(data_frame) - forward_hours):
        row = data_frame.iloc[i]
        signal = rule_strategy.generate_signal(row)

        if signal == 0:
            continue

        probability = lr_strategy.get_probability(row, signal)
        entry_price = row["close"]
        return_pct = _execute_trade(data_frame, i, entry_price, signal)
        trades.append((return_pct, probability))

    return trades if len(trades) > 0 else None


def group_trades_by_interval(trades):
    """Group trades by probability interval.
    
    Returns a dictionary with interval labels as keys and lists of returns as values.
    """
    results = {f"{low}-{high}": [] for low, high in INTERVALS}

    for return_pct, probability in trades:
        for low, high in INTERVALS:
            if low <= probability < high:
                results[f"{low}-{high}"].append(return_pct)
                break

    # Replace empty lists with None
    return {label: trades if len(trades) > 0 else None 
            for label, trades in results.items()}