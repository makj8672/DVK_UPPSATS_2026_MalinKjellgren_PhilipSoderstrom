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
        self.total_return = sum(trades)
        self.avg_return = self.total_return / self.total_trades
        self.best_trade = max(trades)
        self.worst_trade = min(trades)
        self.sharpe_ratio = self._sharpe_ratio(trades)
        self.max_drawdown = self._max_drawdown(trades)

    def _sharpe_ratio(self, trades):
        returns = pd.Series(trades)
        if returns.std() == 0:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(252)

    def _max_drawdown(self, trades):
        cumulative = pd.Series(trades).cumsum()
        peak = cumulative.cummax()
        drawdown = cumulative - peak
        return drawdown.min()

    def print_results(self):
        print(f"\n--- Backtest-resultat: {self.strategy_name} ---")
        print(f"Antal trades:        {self.total_trades}")
        print(f"Vinnande trades:     {self.winning_trades} ({self.win_rate:.1f}%)")
        print(f"Genomsnittlig trade: {self.avg_return:.2f}%")
        print(f"Total avkastning:    {self.total_return:.2f}%")
        print(f"Bästa trade:         {self.best_trade:.2f}%")
        print(f"Sämsta trade:        {self.worst_trade:.2f}%")
        print(f"Sharpe ratio:        {self.sharpe_ratio:.2f}")
        print(f"Max drawdown:        {self.max_drawdown:.2f}%")

    @classmethod
    def print_interval_table(cls, interval_results):
        """Print a summary table of backtest results per probability interval."""
        print(f"\n{'Intervall':<12} {'Trades':<8} {'Win rate':<10} {'Avg return':<12} {'Sharpe':<8} {'Drawdown':<10}")
        print("-" * 60)
        
        for label, trades in interval_results.items():
            if trades is None:
                print(f"{label:<12} {'Inga trades':<8}")
                continue
            
            result = cls(trades, label)
            print(f"{label:<12} {result.total_trades:<8} {result.win_rate:<10.1f} {result.avg_return:<12.2f} {result.sharpe_ratio:<8.2f} {result.max_drawdown:<10.2f}")