import pandas as pd
import numpy as np

from src.core.strategy import Strategy

def choppiness(high: pd.Series,
               low: pd.Series,
               close: pd.Series,
               length: int = 14) -> pd.Series:
    """
    Choppiness Index (CHOP)

    Parameters
    ----------
    high, low, close : pd.Series
    length           : look-back period (default 14)

    Returns
    -------
    pd.Series of CHOP values in the 0-100 range
    """
    # True Range (vectorised)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    sum_tr   = tr.rolling(length).sum()
    price_range = high.rolling(length).max() - low.rolling(length).min()

    # Avoid divide-by-zero if range is zero
    chop = 100 * np.log10(sum_tr / price_range.replace(0, np.nan)) / np.log10(length)
    return chop

class BarnabasEWMA(Strategy):
    def __init__(self, initial_capital=10000, fast=20, slow=200):
        super().__init__(
            initial_capital=initial_capital,
            author_name="Barnabas",
            strategy_name="Multi EWMA Crossover (Choppy)",
            description="Goes long when the fast EWMA crosses above the slow EWMA, and exits when it crosses below. Prevents trades when the market is choppy."
        )
        self.prices = []
        self.fast = fast
        self.slow = slow
        self.fast2 = fast * 2
        self.slow2 = slow * 2
        self.last_signal = 'hold'

    def process_bar(self, bar):
        """
        Process each bar of data.
        This is where you implement your strategy logic.
        
        Args:
            bar: Dictionary containing 'time', 'close', and 'volume' data
        """
        self.current_bar = bar
        
        # Add your strategy logic here
        # For example:
        # if self.current_bar['close'] > self.previous_close * (1 + self.threshold):
        #     self.last_signal = 'buy'
        # elif self.current_bar['close'] < self.previous_close * (1 - self.threshold):
        #     self.last_signal = 'sell'
        # else:
        #     self.last_signal = 'hold'

    def get_signal(self):
        """
        Return the current trading signal.
        Must return one of: 'buy', 'sell', 'hold'
        """
        # Add your signal generation logic here
        return 'hold'

    def get_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Vectorized version of signal generation.
        Override this if you want to implement a more efficient vectorized version.
        """
        fast_ma = df['close'].ewm(span=self.fast).mean()
        slow_ma = df['close'].ewm(span=self.slow).mean()
        fast2_ma = df['close'].ewm(span=self.fast2).mean()
        slow2_ma = df['close'].ewm(span=self.slow2).mean()
        signals = pd.Series('hold', index=df.index)
        choppy = choppiness(df['high'], df['low'], df['close'], length=self.slow2)
        signals[(fast_ma > slow_ma) & (fast2_ma > slow2_ma) & (choppy < 60)] = 'buy'
        signals[(fast_ma < slow_ma)] = 'sell'
        signals.iloc[:self.slow-1] = 'hold'

        signals = signals.shift(1).fillna('hold')
        return signals