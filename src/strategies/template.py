"""
Template for creating a new trading strategy.
Copy this file and modify it to implement your strategy.
"""

from src.core.strategy import Strategy

class TemplateStrategy(Strategy):
    def __init__(self, initial_capital=10000):
        super().__init__(
            initial_capital=initial_capital,
            author_name="YOUR_NAME",  # Replace with your name
            strategy_name="YOUR_STRATEGY_NAME",  # Replace with your strategy name
            description="DESCRIBE YOUR STRATEGY HERE"  # Replace with your strategy description
        )
        
        # Add any strategy-specific initialization here
        # For example:
        # self.lookback_period = 20
        # self.threshold = 0.02

    def get_signals(self, df):
        """
        Vectorized version of signal generation.
        Override this if you want to implement a more efficient vectorized version.
        """
        return super().get_signals(df) 