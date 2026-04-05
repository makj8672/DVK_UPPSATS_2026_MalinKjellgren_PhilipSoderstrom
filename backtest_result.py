# backtest_result.py
# Holds the results of a backtest

#TODO: kanske ett alternativ till testning. En klass som håller på resultaten av backtesten

import numpy as np
import pandas as pd

class BacktestResult:
    def __init__(self, trades, strategy_name):
        self.strategy_name = strategy_name
        self.total_trades = len(trades)
        self.winning_trades = sum(1 for t in trades if t > 0)
        self.win_rate = self.winning_trades / self.total_trades * 100
        self.total_return = self._total_return(trades)
        self.avg_return = self._avg_return(trades)
        self.best_trade = max(trades)
        self.worst_trade = min(trades)
        self.sharpe_ratio = self._sharpe_ratio(trades)
        self.max_drawdown = self._max_drawdown(trades)
        self.expectancy = self._expectancy(trades)
        self.profit_factor = self._profit_factor(trades)

    def _total_return(self, trades):
        """Calculate compounded total return."""
        equity = 100.0
        for trade in trades:
            equity *= (1 + trade / 100)
        return equity - 100.0
    
    def _avg_return(self, trades):
        """Calculate geometric mean return per trade."""
        equity = 100.0
        for trade in trades:
            equity *= (1 + trade / 100)
        return (equity / 100) ** (1 / len(trades)) * 100 - 100
    
    def _sharpe_ratio(self, trades):
        returns = pd.Series(trades)
        if returns.std() == 0:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(252)

    def _max_drawdown(self, trades):
        """Calculate max drawdown as percentage of peak equity."""
        equity = 100.0  # Start with 100 as base
        peak = equity
        max_dd = 0.0

        for trade in trades:
            equity *= (1 + trade / 100)  # compound returns
            if equity > peak:
                peak = equity
            drawdown = (equity - peak) / peak * 100
            if drawdown < max_dd:
                max_dd = drawdown

        return max_dd
    
    def _expectancy(self, trades):
        """Expectancy = (WR x avg_gain) - (LR x avg_loss)"""
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        win_rate = len(wins) / len(trades)
        loss_rate = len(losses) / len(trades)
        avg_gain = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0
        return (win_rate * avg_gain) - (loss_rate * avg_loss)

    def _profit_factor(self, trades):
        """Profit factor = gross profit / gross loss"""
        gross_profit = sum(t for t in trades if t > 0)
        gross_loss = abs(sum(t for t in trades if t < 0))
        if gross_loss == 0:
            return float('inf')
        return gross_profit / gross_loss

    def print_results(self):
        print(f"\n--- Backtest result: {self.strategy_name} ---")
        print(f"Total trades:        {self.total_trades}")
        print(f"Winning trades:      {self.winning_trades} ({self.win_rate:.1f}%)")
        print(f"Average trade:       {self.avg_return:.2f}%")
        print(f"Total return:        {self.total_return:.2f}%")
        print(f"Best trade:          {self.best_trade:.2f}%")
        print(f"Worst trade:         {self.worst_trade:.2f}%")
        print(f"Sharpe ratio:        {self.sharpe_ratio:.2f}")
        print(f"Max drawdown:        {self.max_drawdown:.2f}%")
        print(f"Expectancy:          {self.expectancy:.2f}%")
        print(f"Profit factor:       {self.profit_factor:.2f}")

    @classmethod
    def print_interval_table(cls, interval_results):
        """Print a summary table of backtest results per probability interval."""
        print(f"\n{'Intervall':<12} {'Trades':<8} {'Win rate':<10} {'Expectancy':<12} {'Profit F':<10} {'Avg return':<12} {'Sharpe':<8} {'Drawdown':<10}")
        print("-" * 60)
        
        for label, trades in interval_results.items():
            if trades is None:
                print(f"{label:<12} {'Inga trades':<8}")
                continue
            
            result = cls(trades, label)
            print(f"{label:<12} {result.total_trades:<8} {result.win_rate:<10.1f} {result.expectancy:<12.2f} {result.profit_factor:<10.2f} {result.avg_return:<12.2f} {result.sharpe_ratio:<8.2f} {result.max_drawdown:<10.2f}")