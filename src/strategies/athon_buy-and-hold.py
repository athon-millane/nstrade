"""
Buy and Hold strategy for Bitcoin.
Buys at the first available bar and holds the position until the end.
"""

from src.core.strategy import Strategy
import pandas as pd

class BuyAndHoldStrategy(Strategy):
    def __init__(self, initial_capital=10000):
        super().__init__(
            initial_capital=initial_capital,
            author_name="Athon",
            strategy_name="Buy and Hold",
            description="Buys Bitcoin at the first bar and holds the position for the entire backtest period."
        )

    def get_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series('hold', index=df.index)
        signals.iloc[0] = 'buy'
        return signals