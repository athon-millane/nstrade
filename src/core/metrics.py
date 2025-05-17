import numpy as np
import pandas as pd

def sharpe_ratio(returns, periods_per_year=8760):
    """
    Calculate the annualized Sharpe ratio for hourly returns.
    Assumes risk-free rate is 0.
    """
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret == 0 or np.isnan(std_ret):
        return 0.0
    sharpe = (mean_ret * periods_per_year) / (std_ret * np.sqrt(periods_per_year))
    return sharpe

def total_return(equity_curve):
    """
    Total return over the period.
    """
    if equity_curve.iloc[0] == 0 or np.isnan(equity_curve.iloc[0]):
        return 0.0
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

def n_trades(trades):
    """
    Number of completed trades.
    """
    return len(trades)

def win_rate(trades):
    """
    Proportion of trades with positive PnL.
    """
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t['pnl'] > 0)
    return wins / len(trades)

def max_drawdown(equity_curve):
    """
    Maximum drawdown (as a positive fraction, e.g. 0.2 for 20%).
    """
    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    # Avoid division by zero
    peak[peak == 0] = 1
    drawdown = (peak - equity) / peak
    return np.max(drawdown)

def annualized_return(equity_curve, periods_per_year=8760):
    """
    Annualized return based on start/end equity and number of periods.
    """
    n_periods = len(equity_curve) - 1
    if n_periods == 0 or equity_curve.iloc[0] == 0 or np.isnan(equity_curve.iloc[0]):
        return 0.0
    total_ret = equity_curve.iloc[-1] / equity_curve.iloc[0]
    ann_ret = total_ret ** (periods_per_year / n_periods) - 1
    return ann_ret

def r_squared_equity_curve(equity_curve):
    """
    Calculate R-squared of equity curve against a linear regression line.
    Higher values indicate more consistent returns (closer to a straight line).
    Returns a tuple of (r_squared, slope, intercept) where r_squared is between 0 and 1.
    """
    if len(equity_curve) <= 2:
        return 0.0, 0.0, 0.0
    
    # Create x values (time indices)
    x = np.arange(len(equity_curve))
    y = np.array(equity_curve)
    
    # Calculate linear regression
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    
    # Calculate R-squared
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - y_pred)**2)
    
    if ss_total == 0:
        return 0.0, slope, intercept
    
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared, slope, intercept


def rolling_sharpe_ratio(equity_curve, window=720, periods_per_year=8760):
    """
    Calculate the rolling Sharpe ratio (annualized) on log returns.
    Returns a pandas Series aligned with the equity curve.
    """
    equity_curve = pd.Series(equity_curve)
    # Use pct_change instead of log returns for better numerical stability
    returns = equity_curve.pct_change().fillna(0)
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    # Annualize
    rolling_sharpe = (rolling_mean * periods_per_year) / (rolling_std * np.sqrt(periods_per_year))
    return rolling_sharpe.fillna(0)

def unrealized_drawdown_series(equity_curve):
    """
    Returns a pandas Series of the running (unrealized) drawdown at each time step.
    """
    equity = pd.Series(equity_curve)
    peak = equity.cummax()
    # Avoid division by zero
    peak[peak == 0] = 1
    drawdown = (peak - equity) / peak
    return drawdown.fillna(0)

def realized_drawdown_series(equity_curve, trades):
    """
    Returns a pandas Series with realized drawdown at each trade exit (sell).
    The value is NaN except at trade exits, where it is the max drawdown during that trade.
    """
    equity = pd.Series(equity_curve)
    realized_dd = pd.Series(index=equity.index, dtype=float)
    
    for trade in trades:
        entry_idx = trade['entry_idx']
        exit_idx = trade['exit_idx']
        
        # Skip if indices are out of bounds
        if entry_idx < 0 or exit_idx < 0 or entry_idx >= len(equity) or exit_idx >= len(equity):
            continue
            
        # Get the equity curve for this trade
        trade_equity = equity.iloc[entry_idx:exit_idx+1]
        if len(trade_equity) == 0:
            continue
            
        # Calculate drawdown for this trade
        peak = trade_equity.cummax()
        # Avoid division by zero
        peak[peak == 0] = 1
        dd = (peak - trade_equity) / peak
        realized_dd.iloc[exit_idx] = dd.max()
        
    return realized_dd 