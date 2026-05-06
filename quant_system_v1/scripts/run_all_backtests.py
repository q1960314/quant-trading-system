"""
8-strategy comprehensive backtest on 2024 Q4.
"""
import os, sys, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
import pandas as pd

from backtest.vectorized_engine import VectorizedBacktestEngine
from backtest.report import BacktestReport
from strategy import StrategyRegistry
from config.settings import INIT_CAPITAL

for mod in ['board_strategy','shrink_volume','sector_rotation','ma_breakout_wrapper',
            'first_board_strategy','dragon_rebound_strategy','broken_board_rebound_strategy','ml_strategy']:
    __import__(f'strategy.{mod}')

START = "2024-10-08"
END = "2024-12-31"
CAPITAL = INIT_CAPITAL


def main():
    strategies = StrategyRegistry.list_all()
    print(f"Comprehensive Backtest: {len(strategies)} strategies | {START} ~ {END}")
    print("=" * 70)

    # Load data once
    engine = VectorizedBacktestEngine(initial_capital=CAPITAL, start_date=START, end_date=END,
                                      max_hold_stocks=5, single_stock_position=0.2)
    print("Loading data...", flush=True)
    engine.load_data()
    engine.constraint_matrix = engine.build_constraint_matrix()
    print(f"Data: {len(engine.codes)} stocks x {len(engine.dates)} days\n")

    results = {}
    for name in strategies:
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            strategy = StrategyRegistry.get(name)
            result = engine._run_with_data(strategy)
            elapsed = time.time() - t0
            r = {
                'final_capital': result.final_capital,
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
            }
            results[name] = r
            print(f"R={r['total_return']*100:+.2f}% Sharpe={r['sharpe_ratio']:.2f} Trades={r['total_trades']} ({elapsed:.1f}s)")
        except Exception as e:
            print(f"FAIL: {e}")
            results[name] = {'total_return': 0, 'sharpe_ratio': -9, 'total_trades': 0, 'error': str(e)}

    print("\n" + "=" * 70)
    print(f"{'Strategy':<18} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7} {'Trades':>7} {'WinRate':>7}")
    print("-" * 70)
    for name, r in sorted(results.items(), key=lambda x: x[1].get('sharpe_ratio', -9), reverse=True):
        if 'error' in r:
            print(f"{name:<18} ERROR: {r['error'][:35]}")
        else:
            print(f"{name:<18} {r['total_return']*100:>7.2f}% {r['sharpe_ratio']:>6.2f} "
                  f"{r['max_drawdown']*100:>6.2f}% {r['total_trades']:>6} {r['win_rate']:>6.1f}%")
    print("=" * 70)

    # Save CSV
    df = pd.DataFrame(results).T
    out = f"backtest_8strategy_{START}_{END}.csv"
    df.to_csv(out)
    print(f"\nSaved: {out}")

    # Generate HTML for best
    best_name = max(results.items(), key=lambda x: x[1].get('sharpe_ratio', -9))[0]
    best = StrategyRegistry.get(best_name)
    print(f"\nGenerating HTML report for: {best_name}")
    try:
        result = engine._run_with_data(best)
        report = BacktestReport(result, best_name, output_dir=".")
        path = report.generate()
        print(f"Report: {path}")
    except Exception as e:
        print(f"Report failed: {e}")


if __name__ == '__main__':
    main()
