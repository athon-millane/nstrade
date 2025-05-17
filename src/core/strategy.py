from typing import Any, Dict, Optional
import pandas as pd

class Strategy:
    def __init__(self, 
                 initial_capital: float = 10000,
                 author_name: Optional[str] = None,
                 strategy_name: Optional[str] = None,
                 description: Optional[str] = None):
        """
        Initialize a trading strategy.
        
        Args:
            initial_capital: Starting capital for the strategy
            author_name: Name of the strategy author
            strategy_name: Name of the strategy
            description: Description of the strategy
        """
        self.initial_capital = initial_capital
        self.position = 0  # 0 = out, 1 = long
        self.entry_price = None
        self.trades = []  # List of dicts: {'entry': ..., 'exit': ..., 'pnl': ...}
        self.current_bar = None
        
        # Metadata
        self.author_name = author_name
        self.strategy_name = strategy_name
        self.description = description

    def validate(self) -> bool:
        """
        Validates the strategy implementation.
        Raises ValueError if validation fails.
        
        Returns:
            bool: True if validation passes
        """
        if not self.author_name:
            raise ValueError("Strategy must have an author_name")
        if not self.strategy_name:
            raise ValueError("Strategy must have a strategy_name")
        if not self.description:
            raise ValueError("Strategy must have a description")
        return True

    def get_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Vectorized version of signal generation.
        Returns a Series of signals ('buy', 'sell', 'hold') for the entire dataset.
        By default, implements the same logic as get_signal() but for all bars at once.
        """
        return pd.Series('hold', index=df.index) 