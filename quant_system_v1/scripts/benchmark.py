"""Performance benchmark: CSV vs DuckDB, loading + backtest."""
import sys, os, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')

from backtest.vectorized_engine import VectorizedBacktestEngine
import strategy.board_strategy
from strategy import StrategyRegistry

print("=== Performance Benchmark ===\n")

# Benchmark 1: Data Loading
for label, use_db in [("DuckDB", True), ("CSV", False)]:
    t0 = time.time()
    e = VectorizedBacktestEngine(start_date='2024-10-08', end_date='2024-10-31')
    e.load_data(use_duckdb=use_db)
    t = time.time() - t0
    print(f"{label} load: {t:.1f}s ({len(e.codes)} stocks x {len(e.dates)} days)")

# Benchmark 2: Full Backtest
print()
for label, use_db in [("DuckDB", True), ("CSV", False)]:
    e = VectorizedBacktestEngine(initial_capital=5000, start_date='2024-10-08',
                                  end_date='2024-10-31', max_hold_stocks=5)
    e.load_data(use_duckdb=use_db)
    e.constraint_matrix = e.build_constraint_matrix()
    s = StrategyRegistry.get('打板策略')
    t0 = time.time()
    r = e._run_with_data(s)
    t = time.time() - t0
    print(f"{label} backtest: {t:.1f}s (Ret={r.total_return*100:.2f}%, Trades={r.total_trades})")

print("\n=== Done ===")
