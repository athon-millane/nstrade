"""
Donchian Channel Breakout
"""

from src.core.strategy import Strategy
import pandas as pd

AUTHOR_NAME = "Andy"

class DonchianChannelBreakoutStrategy(Strategy):
    def __init__(self, 
                 initial_capital: float = 10000, 
                 window: int = 20):
        """
        Initialize the Donchian Channel Breakout Strategy.
        
        Args:
            initial_capital: Starting capital for the strategy
            window: Lookback window to determine channel boundaries
        """
        super().__init__(
            initial_capital=initial_capital,
            author_name=AUTHOR_NAME,
            strategy_name=f"Donchian Channel Breakout {window}",
            description=f"Goes long when price breaks above the {window}-period upper Donchian channel, and exits when it breaks below the lower channel."
        )
        self.window = window

    def get_signals(self, df: pd.DataFrame) -> pd.Series:
        # Calculate upper and lower Donchian channels
        upper_channel = df['high'].rolling(window=self.window).max()
        lower_channel = df['low'].rolling(window=self.window).min()
        
        # Initialize signals
        signals = pd.Series('hold', index=df.index)
        
        # Buy when price breaks above upper channel
        buy_condition = df['close'] > upper_channel.shift(1)
        signals[buy_condition] = 'buy'
        
        # Sell when price breaks below lower channel
        sell_condition = df['close'] < lower_channel.shift(1)
        signals[sell_condition] = 'sell'
        
        # No signals can be generated until we have enough data
        signals.iloc[:self.window] = 'hold'
        
        # Shift signals to avoid look-ahead bias
        signals = signals.shift(1).fillna('hold')
        
        return signals
