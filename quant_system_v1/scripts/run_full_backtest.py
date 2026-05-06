"""
Comprehensive backtest: all 8 strategies on same period, compare + report.

Usage: python scripts/run_full_backtest.py
"""
import os, sys, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

import pandas as pd
from backtest.vectorized_engine import VectorizedBacktestEngine
from backtest.report import BacktestReport
from strategy import StrategyRegistry
from config.settings import INIT_CAPITAL

# Import all strategies to register them
import strategy.board_strategy
import strategy.shrink_volume
import strategy.sector_rotation
import strategy.ma_breakout_wrapper
import strategy.first_board_strategy
import strategy.dragon_rebound_strategy
import strategy.broken_board_rebound_strategy
import strategy.ml_strategy

START = "2024-10-08"
END = "2024-12-31"
CAPITAL = INIT_CAPITAL


def run_all():
    strategies = StrategyRegistry.list_all()
    print(f"\n{'='*60}")
    print(f"  COMPREHENSIVE BACKTEST: {len(strategies)} strategies")
    print(f"  Period: {START} ~ {END}  |  Capital: {CAPITAL:,}")
    print(f"{'='*60}\n")

    results = {}
    engine = VectorizedBacktestEngine(
        initial_capital=CAPITAL, start_date=START, end_date=END,
        max_hold_stocks=5, single_stock_position=0.2,
    )
    engine.load_data()
    engine.constraint_matrix = engine.build_constraint_matrix()

    for i, name in enumerate(strategies):
        print(f"[{i+1}/{len(strategies)}] {name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            strategy = StrategyRegistry.get(name)
            result = engine._run_with_data(strategy)
            elapsed = time.time() - t0
            results[name] = {
                'final_capital': result.final_capital,
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
            }
            print(f"Return={result.total_return*100:+.2f}% Sharpe={result.sharpe_ratio:.2f} Trades={result.total_trades} ({elapsed:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            results[name] = {'total_return': 0, 'sharpe_ratio': 0, 'total_trades': 0, 'error': str(e)}

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Strategy':<16} {'Return':>8} {'Annual':>8} {'Sharpe':>7} {'MaxDD':>7} {'Trades':>7} {'WinRate':>7}")
    print(f"{'-'*80}")
    for name, r in results.items():
        if 'error' in r:
            print(f"{name:<16} {'ERROR: '+r['error'][:40]}")
        else:
            print(f"{name:<16} {r['total_return']*100:>7.2f}% {r['annual_return']*100:>7.2f}% "
                  f"{r['sharpe_ratio']:>6.2f} {r['max_drawdown']*100:>6.2f}% {r['total_trades']:>6} "
                  f"{r['win_rate']:>6.1f}%")
    print(f"{'='*80}")

    # Export to CSV
    df_result = pd.DataFrame(results).T
    df_result.index.name = 'strategy'
    out = f"backtest_comparison_{START}_{END}.csv"
    df_result.to_csv(out)
    print(f"\nResults saved to {out}")

    # Generate HTML report for best strategy
    best = max(results.items(), key=lambda x: x[1].get('sharpe_ratio', -999))
    print(f"\nBest strategy by Sharpe: {best[0]} (Sharpe={best[1]['sharpe_ratio']:.2f})")
    return results


if __name__ == '__main__':
    run_all()
