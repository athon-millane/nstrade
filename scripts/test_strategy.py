#!/usr/bin/env python3
"""
CLI tool for testing trading strategies.
Usage: python scripts/test_strategy.py path/to/strategy.py
"""

import argparse
from pathlib import Path
import importlib.util
import pandas as pd
from datetime import datetime
import sys

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.backtest import run_backtest
from src.core.strategy import Strategy

def load_strategy(strategy_path: str) -> type[Strategy]:
    """Load a strategy class from a Python file."""
    strategy_path = Path(strategy_path)
    if not strategy_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {strategy_path}")
        
    # Load the module
    spec = importlib.util.spec_from_file_location("strategy", strategy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find the strategy class
    strategy_class = next(
        (obj for name, obj in module.__dict__.items() 
         if isinstance(obj, type) and issubclass(obj, Strategy) and obj != Strategy),
        None
    )
    
    if not strategy_class:
        raise ValueError("No strategy class found in file")
        
    return strategy_class

def test_strategy(strategy_path: str, return_results: bool = False):
    """
    Test a strategy on development and holdout periods.
    
    Args:
        strategy_path: Path to the strategy file
        return_results: If True, return the results instead of printing them
        
    Returns:
        If return_results is True, returns (dev_results, holdout_results)
        Otherwise returns None
    """
    # Load strategy
    strategy_class = load_strategy(strategy_path)
    
    # Validate strategy
    strategy = strategy_class()
    try:
        strategy.validate()
    except ValueError as e:
        print(f"Validation failed: {e}")
        return None if not return_results else (None, None)
    
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "btc_hour.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    
    # Run backtest on development period
    if not return_results:
        print("\nTesting on development period (2011-2024)...")
    dev_results = run_backtest(
        strategy_class, 
        df,
        start_date="2011-01-01",
        end_date="2024-12-31"
    )
    
    if not return_results:
        print(f"Sharpe Ratio: {dev_results['sharpe']:.2f}")
        print(f"Total Return: {dev_results['total_return']*100:.2f}%")
        print(f"Max Drawdown: {dev_results['max_drawdown']*100:.2f}%")
        print(f"Number of Trades: {dev_results['n_trades']}")
        print(f"Win Rate: {dev_results['win_rate']*100:.2f}%")
        print(f"R^2: {dev_results['r_squared']:.2f}")
    
    # Run backtest on holdout period
    if not return_results:
        print("\nTesting on holdout period (2025)...")
    holdout_results = run_backtest(
        strategy_class,
        df,
        start_date="2025-01-01"
    )
    
    if not return_results:
        print(f"Sharpe Ratio: {holdout_results['sharpe']:.2f}")
        print(f"Total Return: {holdout_results['total_return']*100:.2f}%")
        print(f"Max Drawdown: {holdout_results['max_drawdown']*100:.2f}%")
        print(f"Number of Trades: {holdout_results['n_trades']}")
        print(f"Win Rate: {holdout_results['win_rate']*100:.2f}%")
        print(f"R^2: {holdout_results['r_squared']:.2f}")
        
    if return_results:
        return dev_results, holdout_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trading strategy")
    parser.add_argument("strategy_path", help="Path to strategy file")
    args = parser.parse_args()
    test_strategy(args.strategy_path) 