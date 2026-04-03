#backtest.py
# Compare both strategies across probability intervals

stop_loss_pct = 1.0
take_profit_pct = 2.0
forward_hours = 24

INTERVALS = [
    (0.0, 0.2, "short"),
    (0.2, 0.4, "short"),
    (0.4, 0.6, "hold"),
    (0.6, 0.8, "long"),
    (0.8, 1.0, "long")
]

def run_backtest_for_interval(strategy, data_frame, low, high, direction):
    """Run backtest for a specific probability interval."""
    test_data = data_frame.iloc[int(len(data_frame) * 0.8):]
    trades = []

    for i in range(len(test_data) - forward_hours):
        row = test_data.iloc[i]
        probability = strategy.get_probability(row)

        if probability is None or not (low <= probability < high):
            continue

        if direction == "hold":
            continue

        entry_price = row["close"]
        exit_price = entry_price

        for h in range(1, forward_hours + 1):
            if i + h >= len(test_data):
                break
            current_price = test_data.iloc[i + h]["close"]

            if direction == "long":
                current_return = (current_price - entry_price) / entry_price * 100
            else:  # short
                current_return = (entry_price - current_price) / entry_price * 100

            if current_return <= -stop_loss_pct:
                exit_price = current_price
                break
            if current_return >= take_profit_pct:
                exit_price = current_price
                break
            if h == forward_hours:
                exit_price = current_price

        if direction == "long":
            return_pct = (exit_price - entry_price) / entry_price * 100
        else:  # short
            return_pct = (entry_price - exit_price) / entry_price * 100

        trades.append(return_pct)

    return trades if len(trades) > 0 else None


def run_backtest_all_intervals(strategy, data_frame):
    """Run backtest for all probability intervals and return results."""
    results = {}
    for low, high, direction in INTERVALS:
        label = f"{low}-{high}"
        trades = run_backtest_for_interval(strategy, data_frame, low, high, direction)
        results[label] = trades
    return results


def run_backtest(strategy, data_frame):
    """Original backtest for rule-based strategy."""
    test_data = data_frame.iloc[int(len(data_frame) * 0.8):]
    trades = []

    for i in range(len(test_data) - forward_hours):
        row = test_data.iloc[i]
        signal = strategy.generate_signal(row)

        if signal == 1:
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

    return trades if len(trades) > 0 else None