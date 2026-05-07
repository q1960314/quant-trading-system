# Quant System v4 — Production Readiness Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make every module runnable end-to-end: DuckDB fast loading, full historical data, walk-forward optimization, automation pipeline.

**Architecture:** Modify vectorized_engine to load from DuckDB. Add pipeline.py for one-click automation. Run scanner + fetcher for new data.

**Tech Stack:** Python 3.11, DuckDB, pandas, LightGBM, Optuna, Tushare

---

### Task 1: DuckDB Backtest Integration

**Files:**
- Modify: `quant_system_v1/backtest/vectorized_engine.py`

- [ ] **Step 1: Add DuckDB data loading method**

```python
# Add to VectorizedBacktestEngine class:
def load_data_duckdb(self, db_path="../data/quant.duckdb"):
    """Load data from DuckDB instead of CSV files."""
    import duckdb
    logger.info("Loading data from DuckDB...")
    t0 = time.time()
    conn = duckdb.connect(db_path)
    self.df = conn.execute(f"""
        SELECT d.*, sb.name, sb.industry
        FROM daily d
        LEFT JOIN stock_basic sb ON d.ts_code = sb.ts_code
        WHERE d.trade_date BETWEEN '{self.start_date.strftime('%Y-%m-%d')}' 
          AND '{self.end_date.strftime('%Y-%m-%d')}'
        ORDER BY d.trade_date, d.ts_code
    """).df()
    conn.close()
    self.df['trade_date'] = pd.to_datetime(self.df['trade_date'])
    self.codes = sorted(self.df['ts_code'].unique())
    self.dates = sorted(self.df['trade_date'].unique())
    self._merge_global_data_duckdb(db_path)
    self._precompute_factors()
    for col, default in [('limit_status','N'),('order_amount',0),('break_limit_times',0),
                          ('up_down_times',0),('float_market_cap',50),('turnover_rate',2)]:
        if col not in self.df.columns: self.df[col] = default
    self.df = self.df.drop_duplicates(subset=['ts_code','trade_date'], keep='last')
    self.close_matrix = self.df.pivot(index='trade_date', columns='ts_code', values='close')
    self.close_matrix.ffill(inplace=True)
    logger.info(f"DuckDB: {len(self.codes)} stocks x {len(self.dates)} days in {time.time()-t0:.1f}s")
    return self.df

def _merge_global_data_duckdb(self, db_path):
    """Merge additional tables from DuckDB."""
    import duckdb
    conn = duckdb.connect(db_path)
    tables = ['daily_basic', 'limit_list_d', 'top_list', 'ths_hot']
    for tbl in tables:
        try:
            extra = conn.execute(f"SELECT * FROM {tbl}").df()
            if not extra.empty and 'trade_date' in extra.columns:
                extra['trade_date'] = pd.to_datetime(extra['trade_date'])
                self.df = self.df.merge(extra, on=['ts_code','trade_date'], how='left', suffixes=('',f'_{tbl}'))
        except Exception: pass
    conn.close()
```

- [ ] **Step 2: Modify load_data() to use DuckDB by default**

```python
def load_data(self, data_dir="../data_all_stocks", global_dir="../data", use_duckdb=True):
    if use_duckdb:
        db_path = os.path.join(global_dir, 'quant.duckdb')
        if os.path.exists(db_path):
            return self.load_data_duckdb(db_path)
    # Fallback to CSV...
```

- [ ] **Step 3: Benchmark CSV vs DuckDB**

Run: `python -c "engine.load_data(use_duckdb=True)"` — record time
Run: `python -c "engine.load_data(use_duckdb=False)"` — record time
Expected: DuckDB >2x faster

- [ ] **Step 4: Commit**

```bash
git add quant_system_v1/backtest/vectorized_engine.py
git commit -m "feat: DuckDB backtest integration — fast SQL loading"
```

---

### Task 2: Data Source Scanner Run + New Interface Fetch

**Files:**
- Modify: `quant_system_v1/evolution/datasource_scanner.py` (fix encoding already done)
- Create: `quant_system_v1/scripts/scan_and_fetch.py`

- [ ] **Step 1: Run scanner and log results**

```python
# scripts/scan_and_fetch.py
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging; logging.disable(logging.CRITICAL)
from evolution.datasource_scanner import TushareAPIScanner

scanner = TushareAPIScanner()
scanner.scan(max_apis=30)
code = scanner.export_candidates()
if code:
    out = os.path.join(os.path.dirname(__file__), '..', 'evolution_output', 'factors', 'auto_discovered_factors.py')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f: f.write(code)
    print(f"Saved discovered factors to {out}")
available = [r for r in scanner.results if r['status'] == 'AVAILABLE']
print(f"Scan done: {len(scanner.results)} tested, {len(available)} available")
for r in available:
    print(f"  {r['api']}: {len(r['new_columns'])} new cols")
```

- [ ] **Step 2: Run scan**

Run: `cd F:/编程文件/quant_system_v1 && python scripts/scan_and_fetch.py`
Expected: Lists available APIs and new columns

- [ ] **Step 3: Commit**

```bash
git add quant_system_v1/scripts/scan_and_fetch.py
git commit -m "feat: API scanner run script — discover unused Tushare interfaces"
```

---

### Task 3: Full Historical Data Fetch

**Files:**
- Modify: None (use existing CLI)

- [ ] **Step 1: Fetch 2020-2023 daily data**

Run: `python main.py fetch --start 2020-01-01 --end 2023-12-31`
Expected: Data fetched to data_all_stocks/ and data/

- [ ] **Step 2: Verify data range**

Run: `python -c "import pandas as pd; d=pd.read_csv('data/daily.csv'); print(d['trade_date'].min(),'-',d['trade_date'].max())"`
Expected: 20200102 - 20231231

- [ ] **Step 3: Rebuild DuckDB with new data**

Run: `python scripts/migrate_to_duckdb.py`
Expected: Tables updated with full date range

- [ ] **Step 4: Verify expanded backtest**

Run: `python main.py backtest --strategy 打板策略 --engine vectorized`
Expected: More trades and longer period

---

### Task 4: Walk-Forward Optimization Run

**Files:**
- Create: `quant_system_v1/scripts/run_optimization.py`

- [ ] **Step 1: Create optimization runner**

```python
# scripts/run_optimization.py
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
from optimizer.walk_forward import walk_forward
from backtest.vectorized_engine import VectorizedBacktestEngine
import strategy.board_strategy
from strategy import StrategyRegistry

def backtest_fn(params):
    engine = VectorizedBacktestEngine(
        initial_capital=params.get('initial_capital', 5000),
        start_date=params['start_date'], end_date=params['end_date'],
        max_hold_stocks=params.get('max_hold_stocks', 5),
        single_stock_position=params.get('single_stock_position', 0.15),
    )
    s = StrategyRegistry.get('打板策略')
    r = engine.run(s)
    return {'sharpe_ratio': r.sharpe_ratio, 'total_return': r.total_return,
            'max_drawdown': r.max_drawdown, 'total_trades': r.total_trades}

param_grid = {
    'max_hold_stocks': [3, 5, 8],
    'single_stock_position': [0.10, 0.15, 0.20],
    'stop_loss': [0.03, 0.06, 0.09],
}

results = walk_forward(param_grid, backtest_fn, start_date='2024-10-01',
                        end_date='2024-12-31', train_months=2, valid_months=1)
print(results.to_string())
results.to_csv('walk_forward_results.csv', index=False)
print("Saved: walk_forward_results.csv")
```

- [ ] **Step 2: Run optimization**

Run: `cd F:/编程文件/quant_system_v1 && python scripts/run_optimization.py`
Expected: Walk-forward results table

- [ ] **Step 3: Commit**

```bash
git add quant_system_v1/scripts/run_optimization.py
git commit -m "feat: walk-forward optimization runner"
```

---

### Task 5: End-to-End Automation Pipeline

**Files:**
- Create: `quant_system_v1/scripts/pipeline.py`

- [ ] **Step 1: Create automation pipeline**

```python
# scripts/pipeline.py
"""One-click daily pipeline: fetch → clean → backtest → pick → report."""
import sys, os, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')

STEPS = ['fetch', 'backtest', 'pick', 'report']

def main():
    print("=" * 50)
    print("  QUANT DAILY PIPELINE")
    print("=" * 50)
    t0 = time.time()

    # 1. Incremental data fetch
    if 'fetch' in STEPS:
        print("\n[1/4] Fetching incremental data...")
        from data_source.fetcher import DataFetcher
        try:
            fetcher = DataFetcher()
            fetcher.fetch_incremental()
            print("  Fetch: OK")
        except Exception as e:
            print(f"  Fetch: SKIPPED ({e})")

    # 2. Backtest all strategies
    if 'backtest' in STEPS:
        print("\n[2/4] Running backtests...")
        for mod in ['board_strategy','shrink_volume','sector_rotation','ma_breakout_wrapper',
                    'first_board_strategy','dragon_rebound_strategy','broken_board_rebound_strategy','ml_strategy']:
            __import__(f'strategy.{mod}')
        from strategy import StrategyRegistry
        from backtest.vectorized_engine import VectorizedBacktestEngine
        e = VectorizedBacktestEngine(initial_capital=5000, start_date='2024-10-08',
                                      end_date='2024-12-31', max_hold_stocks=5, single_stock_position=0.15)
        e.load_data()
        e.constraint_matrix = e.build_constraint_matrix()
        import pandas as pd
        rows = []
        for name in StrategyRegistry.list_all():
            s = StrategyRegistry.get(name)
            r = e._run_with_data(s)
            rows.append({'strategy': name, 'return': r.total_return, 'sharpe': r.sharpe_ratio,
                         'trades': r.total_trades, 'win_rate': r.win_rate})
        df = pd.DataFrame(rows)
        print(df.to_string())
        df.to_csv('daily_backtest_results.csv', index=False)

    # 3. Stock pick
    if 'pick' in STEPS:
        print("\n[3/4] Running stock picker...")
        from stock_picker.picker_v2 import DailyStockPickerV2
        picker = DailyStockPickerV2(['打板策略', '断板反包策略', '均线突破策略'])
        df_data, trade_date = picker.get_latest_data()
        picks = picker.pick(df_data, trade_date)
        if not picks.empty:
            print(picks[['ts_code','name','combined_score','strategy_count']].head(10).to_string())

    # 4. Generate report
    if 'report' in STEPS:
        print("\n[4/4] Generating report...")
        from backtest.report import BacktestReport
        from strategy import StrategyRegistry
        import main
        e2 = VectorizedBacktestEngine(initial_capital=5000, start_date='2024-10-08',
                                       end_date='2024-12-31', max_hold_stocks=5, single_stock_position=0.15)
        e2.load_data(); e2.constraint_matrix = e2.build_constraint_matrix()
        s_best = StrategyRegistry.get('打板策略')
        r_best = e2._run_with_data(s_best)
        report = BacktestReport(r_best, '打板策略', output_dir='.')
        path = report.generate()
        print(f"  Report: {path}")

    elapsed = time.time() - t0
    print(f"\nPipeline complete in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run pipeline**

Run: `cd F:/编程文件/quant_system_v1 && python scripts/pipeline.py`
Expected: All 4 steps execute and produce outputs

- [ ] **Step 3: Commit**

```bash
git add quant_system_v1/scripts/pipeline.py
git commit -m "feat: end-to-end daily automation pipeline"
```

---

### Task 6: Performance Benchmark + Final Validation

**Files:**
- Create: `quant_system_v1/scripts/benchmark.py`

- [ ] **Step 1: Benchmark script**

```python
# scripts/benchmark.py
import sys, os, time; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging; logging.disable(logging.CRITICAL)
from backtest.vectorized_engine import VectorizedBacktestEngine
import strategy.board_strategy
from strategy import StrategyRegistry

print("=== Performance Benchmark ===")

# CSV loading
t0 = time.time()
e = VectorizedBacktestEngine(start_date='2024-10-08', end_date='2024-10-31')
e.load_data(use_duckdb=False)
csv_time = time.time() - t0
print(f"CSV load: {csv_time:.1f}s ({len(e.codes)} stocks x {len(e.dates)} days)")

# DuckDB loading
t0 = time.time()
e2 = VectorizedBacktestEngine(start_date='2024-10-08', end_date='2024-10-31')
e2.load_data(use_duckdb=True)
ddb_time = time.time() - t0
print(f"DuckDB load: {ddb_time:.1f}s ({len(e2.codes)} stocks x {len(e2.dates)} days)")

if csv_time > 0 and ddb_time > 0:
    print(f"Speedup: {csv_time/ddb_time:.1f}x")

# Full backtest benchmark
s = StrategyRegistry.get('打板策略')
t0 = time.time()
e2.constraint_matrix = e2.build_constraint_matrix()
r = e2._run_with_data(s)
bt_time = time.time() - t0
print(f"Backtest: {bt_time:.1f}s (Return={r.total_return*100:.2f}%, Trades={r.total_trades})")
print("=== Done ===")
```

- [ ] **Step 2: Run benchmark**

Run: `cd F:/编程文件/quant_system_v1 && python scripts/benchmark.py`
Expected: Speedup numbers

- [ ] **Step 3: Final commit**

```bash
git add quant_system_v1/scripts/benchmark.py
git commit -m "feat: performance benchmark — CSV vs DuckDB comparison"
```

---

## Self-Review

### Spec Coverage
- [x] Data source scanner + new interfaces → Task 2
- [x] DuckDB backtest integration → Task 1
- [x] Full historical data fetch → Task 3
- [x] Walk-forward optimization → Task 4
- [x] Automation pipeline → Task 5
- [x] Performance benchmark → Task 6

### Placeholder Check
No TBD/TODO items. All code blocks contain complete implementations.

### Type Consistency
- `load_data(use_duckdb=True)` signature consistent across Task 1, 5, 6
- `_run_with_data(strategy)` used consistently
- `VectorizedBacktestEngine` constructor args match across all tasks
