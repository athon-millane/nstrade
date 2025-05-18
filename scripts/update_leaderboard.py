#!/usr/bin/env python3
"""
Update the leaderboard with strategy results.
Usage: python scripts/update_leaderboard.py path/to/strategy.py
"""

import argparse
from pathlib import Path
import json
from datetime import datetime, UTC
import sys

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.test_strategy import load_strategy, test_strategy

def update_leaderboard(strategy_path: str):
    """Update the leaderboard with strategy results."""
    # Load strategy
    strategy_class = load_strategy(strategy_path)
    strategy = strategy_class()
    
    # Get strategy metadata
    metadata = {
        "author_name": strategy.author_name,
        "strategy_name": strategy.strategy_name,
        "description": strategy.description,
        "last_updated": datetime.now(UTC).isoformat() + "Z"
    }
    
    # Run backtests
    dev_results, holdout_results = test_strategy(strategy_path, return_results=True)
    
    # Extract metrics
    metadata["development_metrics"] = {
        "sharpe": dev_results["sharpe"],
        "total_return": dev_results["total_return"],
        "max_drawdown": dev_results["max_drawdown"],
        "n_trades": dev_results["n_trades"],
        "win_rate": dev_results["win_rate"]
    }
    
    metadata["holdout_metrics"] = {
        "sharpe": holdout_results["sharpe"],
        "total_return": holdout_results["total_return"],
        "max_drawdown": holdout_results["max_drawdown"],
        "n_trades": holdout_results["n_trades"],
        "win_rate": holdout_results["win_rate"]
    }
    
    # Load current leaderboard
    leaderboard_path = Path(__file__).parent.parent / "data" / "leaderboard.json"
    with open(leaderboard_path) as f:
        leaderboard = json.load(f)
    
    # Update or add strategy
    strategy_key = f"{metadata['author_name']}_{metadata['strategy_name']}"
    updated = False
    
    for i, entry in enumerate(leaderboard["strategies"]):
        if (entry["author_name"] == metadata["author_name"] and 
            entry["strategy_name"] == metadata["strategy_name"]):
            leaderboard["strategies"][i] = metadata
            updated = True
            break
    
    if not updated:
        leaderboard["strategies"].append(metadata)
    
    # Sort by development Sharpe ratio
    leaderboard["strategies"].sort(
        key=lambda x: x["development_metrics"]["sharpe"],
        reverse=True
    )
    
    # Save updated leaderboard
    with open(leaderboard_path, "w") as f:
        json.dump(leaderboard, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update the leaderboard with strategy results")
    parser.add_argument("strategy_path", help="Path to strategy file")
    args = parser.parse_args()
    update_leaderboard(args.strategy_path) 