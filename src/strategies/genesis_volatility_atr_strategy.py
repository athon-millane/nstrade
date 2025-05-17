"""
Volatility ATR Strategy for Bitcoin and Cryptocurrencies.
Uses Average True Range (ATR) for both entry signals and dynamic position sizing.
Implements volatility-based stops and breakout confirmation for better performance.
"""

from src.core.strategy import Strategy
import pandas as pd
import numpy as np
import math

class VolatilityATRStrategy(Strategy):
    def __init__(self, initial_capital=10000, 
                 atr_period=14, 
                 atr_multiplier_entry=1.5,
                 atr_multiplier_stop=2.0,
                 rsi_period=14,
                 rsi_oversold=35,
                 rsi_overbought=70,
                 vol_lookback=30,
                 min_vol_percentile=20,
                 max_risk_per_trade=0.02):  # 2% max risk per trade
        super().__init__(
            initial_capital=initial_capital,
            author_name="Genesis",
            strategy_name="Volatility ATR Strategy",
            description="Dynamic volatility-based strategy using ATR for entries, stops, and position sizing."
        )
        # ATR parameters
        self.atr_period = atr_period
        self.atr_multiplier_entry = atr_multiplier_entry
        self.atr_multiplier_stop = atr_multiplier_stop
        
        # RSI parameters for confirmation
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        # Volatility check
        self.vol_lookback = vol_lookback
        self.min_vol_percentile = min_vol_percentile
        
        # Risk management
        self.max_risk_per_trade = max_risk_per_trade
        
        # Data storage
        self.prices = []
        self.highs = []
        self.lows = []
        self.last_signal = 'hold'
        self.current_atr = None
        self.entry_price = 0
        self.stop_loss = 0
        self.position_size = 0
        
    def calculate_atr(self, high_series, low_series, close_series):
        """Calculate Average True Range"""
        tr1 = high_series - low_series
        tr2 = abs(high_series - close_series.shift(1))
        tr3 = abs(low_series - close_series.shift(1))
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()
        return atr
        
    def calculate_rsi(self, series):
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(self.rsi_period).mean()
        
        # Prevent division by zero
        loss = loss.replace(0, 0.001)
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def is_volatility_suitable(self, atr, close):
        """Check if current volatility is suitable for trading"""
        # Calculate volatility as percentage of price
        vol_pct = (atr / close) * 100
        
        # If we don't have enough history yet, assume volatility is suitable
        if len(self.prices) < self.vol_lookback:
            return True
            
        # Get historical volatility percentiles
        vol_pcts = [(self.calculate_atr(pd.Series(self.highs[-self.vol_lookback:]), 
                                       pd.Series(self.lows[-self.vol_lookback:]), 
                                       pd.Series(self.prices[-self.vol_lookback:])).iloc[-1] / 
                   self.prices[-1]) * 100]
        
        # Calculate percentile of current volatility
        current_percentile = sum(1 for x in vol_pcts if x < vol_pct) / len(vol_pcts) * 100
        
        # Return True if volatility is above our minimum threshold
        return current_percentile >= self.min_vol_percentile
    
    def calculate_position_size(self, entry_price, stop_loss):
        """Calculate position size based on risk management rules"""
        if entry_price == stop_loss:
            return 0
            
        risk_amount = self.initial_capital * self.max_risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0
            
        # Calculate units to buy/sell
        position_size = risk_amount / risk_per_unit
        
        # Round down to avoid exceeding max risk
        return math.floor(position_size)
    
    def process_bar(self, bar):
        self.current_bar = bar
        price = bar['close']
        high = bar['high']
        low = bar['low']
        
        self.prices.append(price)
        self.highs.append(high)
        self.lows.append(low)
        
        # Not enough data yet
        min_data_required = max(self.atr_period, self.rsi_period, self.vol_lookback)
        if len(self.prices) < min_data_required:
            self.last_signal = 'hold'
            return
        
        # Calculate indicators
        price_series = pd.Series(self.prices)
        high_series = pd.Series(self.highs)
        low_series = pd.Series(self.lows)
        
        atr = self.calculate_atr(high_series, low_series, price_series).iloc[-1]
        self.current_atr = atr
        
        rsi = self.calculate_rsi(price_series).iloc[-1]
        
        # Check if volatility is suitable for trading
        volatility_suitable = self.is_volatility_suitable(atr, price)
        
        # Process based on current position
        if self.position == 0:  # No position
            if volatility_suitable:
                # Calculate potential entry and stop levels
                entry_level = price + (atr * self.atr_multiplier_entry)
                stop_level = price - (atr * self.atr_multiplier_stop)
                
                # Calculate position size
                potential_position = self.calculate_position_size(entry_level, stop_level)
                
                # Check for breakout condition
                prev_high = high_series.iloc[-2]
                is_breakout = high > prev_high and rsi < self.rsi_overbought
                
                if is_breakout:
                    self.last_signal = 'buy'
                    self.entry_price = price
                    self.stop_loss = stop_level
                    self.position_size = potential_position
                else:
                    self.last_signal = 'hold'
            else:
                self.last_signal = 'hold'
                
        elif self.position == 1:  # Long position
            # Update stop loss for trailing stop (if price has moved in our favor)
            new_stop = price - (atr * self.atr_multiplier_stop)
            if new_stop > self.stop_loss:
                self.stop_loss = new_stop
            
            # Check if stop loss was hit
            if low <= self.stop_loss:
                self.last_signal = 'sell'
                return
                
            # Check for exit based on RSI overbought or trend reversal
            if rsi > self.rsi_overbought:
                self.last_signal = 'sell'
            else:
                self.last_signal = 'hold'

    def get_signal(self):
        return self.last_signal

    def get_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals for backtesting on a DataFrame"""
        # Calculate technical indicators
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(self.rsi_period).mean()
        loss = loss.replace(0, 0.001)  # Prevent division by zero
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate volatility percentile
        vol_pct = (atr / close) * 100
        vol_percentile = vol_pct.rolling(self.vol_lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )
        
        # Initialize signals
        signals = pd.Series('hold', index=df.index)
        
        # Make sure we have enough data
        min_periods = max(self.atr_period, self.rsi_period, self.vol_lookback)
        
        # Track position and management
        position = 0
        entry_price = 0
        stop_loss = 0
        
        for i in range(min_periods, len(df)):
            idx = df.index[i]
            price = df.loc[idx, 'close']
            high_val = df.loc[idx, 'high']
            low_val = df.loc[idx, 'low']
            atr_val = atr.loc[idx]
            rsi_val = rsi.loc[idx]
            vol_pct_val = vol_percentile.loc[idx] if i >= self.vol_lookback else 50
            
            # Check if we can trade based on volatility
            volatility_suitable = vol_pct_val >= self.min_vol_percentile
            
            # Process based on current position
            if position == 0:  # No position
                if volatility_suitable:
                    # Check for breakout
                    prev_high = df.loc[df.index[i-1], 'high']
                    is_breakout = high_val > prev_high and rsi_val < self.rsi_overbought
                    
                    if is_breakout:
                        signals.loc[idx] = 'buy'
                        position = 1
                        entry_price = price
                        stop_loss = price - (atr_val * self.atr_multiplier_stop)
            
            elif position == 1:  # Long position
                # Update stop loss for trailing stop
                new_stop = price - (atr_val * self.atr_multiplier_stop)
                if new_stop > stop_loss:
                    stop_loss = new_stop
                
                # Check if stop loss was hit
                if low_val <= stop_loss:
                    signals.loc[idx] = 'sell'
                    position = 0
                    continue
                
                # Check for exit based on RSI or trend reversal
                if rsi_val > self.rsi_overbought:
                    signals.loc[idx] = 'sell'
                    position = 0
        
        return signals.shift(1).fillna('hold')
        
    def calculate_expected_return(self):
        """Calculate expected return based on historical performance"""
        if len(self.prices) < 100:
            return 0
            
        # This could be expanded with more sophisticated calculations
        return np.mean(np.diff(self.prices[-100:]) / self.prices[-101:-1]) 