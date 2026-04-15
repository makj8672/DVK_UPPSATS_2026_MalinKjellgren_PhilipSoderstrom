#backtest.py

"""Backtesting utilities for trading strategies.
Compare both strategies across probability intervals

Note: Backtest supports both long and short signals. Current
experiments only generate long signals, but the code is kept generic.
"""

# Backtesting settings (static risk/reward)
stop_loss_pct = 1.0     # Stop loss at -1% 
take_profit_pct = 2.0   # Take profit at +2% 
forward_hours = 24      # Max holding period in hours (H1 bars)

# Probability intervals for grouping trades by model confidence
INTERVALS = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0)
]

def _execute_trade(test_data, i, entry_price, signal): 
    """Execute a single trade and return return %

    - signal == 1 -> long
    - signal == -1 -> short 
    """

    # Confirm signal is valid
    if signal not in (1, -1):
        raise ValueError(f"Invalid signal: {signal}. Expected 1 for long or -1 for short.")
    
    exit_price = entry_price    # Default: exit at last checked bar if no SL/TP hit

    # Look forward up to `forward_hours` bars to find the exit
    for h in range(1, forward_hours + 1):
        
        # Check if we´re at the end of the dataset
        if i + h >= len(test_data):                     
            break

        # Get the price at the current hour
        current_price = test_data.iloc[i + h]["close"]

        # Calculate return percentage based on trade direction (long or short)
        if signal == 1:  
            # Long return: profit when price rises
            current_return = (current_price - entry_price) / entry_price * 100  
        else:  
            # Short return: profit when price falls
            current_return = (entry_price - current_price) / entry_price * 100  

        # Exit if stop-loss or take-profit is hit (or max holding time is reached)
        if current_return <= -stop_loss_pct:
            exit_price = current_price          # Stop-loss hit, exit at current price
            break
        if current_return >= take_profit_pct:
            exit_price = current_price          # Take-profit hit, exit at current price
            break
        if h == forward_hours:
            exit_price = current_price          # Time-based exit, exit at current price
    
    # Convert exit price into signed % return for executed direction
    if signal == 1:  
        return (exit_price - entry_price) / entry_price * 100   # Long return %
    return (exit_price - entry_price) / entry_price * 100       # Short return %


def run_backtest(strategy, data_frame):
    """Backtest a strategy on price series.

    The strategy must implement a `generate_signal(row)` method that returns:
    - 1 for long buy signal
    - -1 for short sell signal
    - 0 for hold signal (no trade)"""

    trades = []    # Create empty list to store trade returns

    # Iterate through entry points (leave room for forward window)
    for i in range(len(data_frame) - forward_hours):
        row = data_frame.iloc[i]                # Get the current row of data
        signal = strategy.generate_signal(row)  # Generate trading signal using the strategy

        # Find buy- or sell signal and execute trade
        if signal in (1,-1):
            entry_price = row["close"]                                      # Enter at close price of signal bar
            return_pct = _execute_trade(data_frame, i, entry_price, signal) # Simulate trade and get return %
            trades.append(return_pct)                                       # Store trade return for metrics

    # Return list of trade returns, or None if no trades were executed
    return trades if len(trades) > 0 else None


def run_backtest_with_probabilities(rule_strategy, lr_strategy, data_frame):
    """Backtest rule entries and attach LR probabilities for interval analysis.
    
    Uses the rule-based streategy to decide direction and
    the LR strategy to score each trade with P(success | direction)
    """

    trades = [] # Create empty list to store (return_pct, probability) tuples for each trade

    # Interate through possible entry points (leave room for the forward window)
    for i in range(len(data_frame) - forward_hours):
        row = data_frame.iloc[i]                        # Current candel (potential entry)
        signal = rule_strategy.generate_signal(row)     # Rule decides direction (1, -1 or 0)

        # Only score/record trades where the rule fires
        if signal == 0:
            continue

        probability = lr_strategy.get_probability(row, signal)              # LR: P( sucess|direction )
        entry_price = row["close"]                                          # Enter at close of the signal bar
        return_pct = _execute_trade(data_frame, i, entry_price, signal)     # Simulate trade return for this direction
        trades.append((return_pct, probability))                            # Store (return, probability) for analysis

    # Return None if no rule-triggered trades appeared
    return trades if len(trades) > 0 else None


def group_trades_by_interval(trades):
    """Group trades by probability interval.
    
    Returns a dictionary with interval labels as keys and lists of returns as values.
    """
    # Prepare output buckets per probability interval
    results = {f"{low}-{high}": [] for low, high in INTERVALS}

    # Assign each trade to only one interval (based on its LR probability)
    for return_pct, probability in trades:
        for low, high in INTERVALS:                         
            if low <= probability < high:                   
                results[f"{low}-{high}"].append(return_pct) 
                break                                       # stop after the first mathching interval

    # Return dictionary (if None -> return empty lists)
    return {label: trades if len(trades) > 0 else None  
            for label, trades in results.items()}           