"""
SMA Crossover 1 strategy for Bitcoin.
Goes long when the fast SMA crosses above the slow SMA, and exits when it crosses below.
"""

from src.core.strategy import Strategy
import pandas as pd

class SMACrossoverStrategy(Strategy):
    def __init__(self, initial_capital=10000, fast=32, slow=140):
        super().__init__(
            initial_capital=initial_capital,
            author_name="Lucien",
            strategy_name="SMA Crossover 1",
            description="Goes long when the 32-period SMA crosses above the 140-period SMA, and exits when it crosses below."
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