import pandas as pd
from typing import Optional

from .metrics import (
    sharpe_ratio, total_return, n_trades, win_rate, max_drawdown, annualized_return,
    rolling_sharpe_ratio, unrealized_drawdown_series, realized_drawdown_series
)

def run_backtest(strategy_class, df: pd.DataFrame, initial_capital: float = 10000,
                start_date: Optional[str] = None, end_date: Optional[str] = None):
    """
    Run a backtest for a trading strategy.
    
    Args:
        strategy_class: The strategy class to test
        df: DataFrame with columns ['time', 'close', 'volume']
        initial_capital: Starting capital for the strategy
        start_date: Optional start date for the backtest (inclusive)
        end_date: Optional end date for the backtest (inclusive)
        
    Returns:
        Dictionary containing backtest results and metrics
    """
    # Local copy in memory for us to work with
    df = df.copy()
    
    # Filter by date if specified
    if start_date:
        df = df[df['time'] >= start_date]
    if end_date:
        df = df[df['time'] <= end_date]
        
    if len(df) == 0:
        raise ValueError("No data available for the specified date range")

    # Initialize strategy
    strat = strategy_class(initial_capital=initial_capital)

    # Get all signals at once
    signals = strat.get_signals(df)
    
    # State machine for position: 0 = out, 1 = long
    positions = pd.Series(0.0, index=df.index)
    current_position = 0.0
    for i, signal in enumerate(signals):
        if current_position == 0.0:
            if signal == 'buy':
                current_position = 1.0
        elif current_position == 1.0:
            if signal == 'sell':
                current_position = 0.0
        positions.iloc[i] = current_position
    
    # Calculate position changes
    position_changes = positions.diff().fillna(positions.iloc[0])
    
    # Identify entry and exit points
    entries = position_changes == 1
    exits = position_changes == -1
    
    # Initialize equity curve and coin holdings with proper dtypes
    equity_curve = pd.Series(initial_capital, index=df.index, dtype=float)
    coins = pd.Series(0.0, index=df.index, dtype=float)
    
    # Track current position and coins
    current_position = 0
    current_coins = 0
    
    # Process each bar
    for i in range(len(df)):
        if entries.iloc[i]:
            # Enter position
            current_position = 1
            current_coins = equity_curve.iloc[i-1] / df['close'].iloc[i]
            coins.iloc[i:] = current_coins
        elif exits.iloc[i]:
            # Exit position
            equity_curve.iloc[i] = current_coins * df['close'].iloc[i]
            current_position = 0
            current_coins = 0
            coins.iloc[i:] = 0
        else:
            equity_curve.iloc[i] = equity_curve.iloc[i-1]
        
        # Update equity curve
        if current_position == 1:
            equity_curve.iloc[i] = current_coins * df['close'].iloc[i]
    
    # Generate trade records
    trades = []
    entry_idx = None
    entry_price = None
    
    for i in range(len(df)):
        if entries.iloc[i]:
            entry_idx = i
            entry_price = df['close'].iloc[i]
        elif exits.iloc[i] and entry_idx is not None:
            exit_price = df['close'].iloc[i]
            pnl = (exit_price - entry_price) * coins.iloc[entry_idx]
            
            trades.append({
                'entry': df['time'].iloc[entry_idx],
                'entry_idx': entry_idx,
                'entry_price': entry_price,
                'exit': df['time'].iloc[i],
                'exit_idx': i,
                'exit_price': exit_price,
                'pnl': pnl
            })
            entry_idx = None
            entry_price = None
    
    # Always close any open position at the end
    if entry_idx is not None or current_position == 1:
        if entry_idx is None:
            # Find the last entry point if we're in position but don't have an entry record
            entry_idx = entries[entries].index[-1] if any(entries) else 0
            entry_price = df['close'].iloc[entry_idx]
        
        exit_price = df['close'].iloc[-1]
        pnl = (exit_price - entry_price) * coins.iloc[entry_idx]
        
        trades.append({
            'entry': df['time'].iloc[entry_idx],
            'entry_idx': entry_idx,
            'entry_price': entry_price,
            'exit': df['time'].iloc[-1],
            'exit_idx': len(df) - 1,
            'exit_price': exit_price,
            'pnl': pnl
        })
        
        # Update final equity
        equity_curve.iloc[-1] = initial_capital + sum(t['pnl'] for t in trades)
    
    # Calculate returns (hourly)
    returns = equity_curve.pct_change().fillna(0).values

    # Metrics
    results = {
        'sharpe': sharpe_ratio(returns),
        'total_return': total_return(equity_curve),
        'n_trades': n_trades(trades),
        'win_rate': win_rate(trades),
        'max_drawdown': max_drawdown(equity_curve),
        'annualized_return': annualized_return(equity_curve),
        'equity_curve': equity_curve,
        'trades': trades,
        'rolling_sharpe': rolling_sharpe_ratio(equity_curve),
        'unrealized_drawdown': unrealized_drawdown_series(equity_curve),
        'realized_drawdown': realized_drawdown_series(equity_curve, trades),
    }
    return results 