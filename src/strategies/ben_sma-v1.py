"""
SMA Crossover 1 strategy for Bitcoin - w/ Ben's Flare.
Goes long when the fast SMA crosses above the slow SMA, and exits when it crosses below.
Updated the fast and slow SMA periods to 30 and 120 respectively.
"""

from src.core.strategy import Strategy
import pandas as pd

class SMACrossoverStrategy(Strategy):
    def __init__(self, initial_capital=10000, fast=30, slow=120):
        super().__init__(
            initial_capital=initial_capital,
            author_name="Ben",
            strategy_name="SMA Crossover 1 - f30s120",
            description="Goes long when the 30-period SMA crosses above the 120-period SMA, and exits when it crosses below."
        )
        self.fast = fast
        self.slow = slow

    def get_signals(self, df: pd.DataFrame) -> pd.Series:
        fast_ma = df['close'].rolling(self.fast).mean()
        slow_ma = df['close'].rolling(self.slow).mean()
        signals = pd.Series('hold', index=df.index)
        signals[fast_ma > slow_ma] = 'buy'
        signals[fast_ma < slow_ma] = 'sell'
        signals.iloc[:self.slow-1] = 'hold'
        signals = signals.shift(1).fillna('hold')
        return signals