"""One-click daily pipeline: fetch → backtest → pick → report."""
import sys, os, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
import pandas as pd


def main():
    print("=" * 50)
    print("  QUANT DAILY PIPELINE")
    print("=" * 50)
    t0 = time.time()

    # 1. Incremental data fetch
    print("\n[1/4] Fetching incremental data...")
    try:
        from data_source.fetcher import DataFetcher
        fetcher = DataFetcher()
        fetcher.fetch_incremental()
        print("  Fetch: OK")
    except Exception as e:
        print(f"  Fetch: SKIPPED ({str(e)[:60]})")

    # 2. Backtest all strategies
    print("\n[2/4] Running backtests...")
    for mod in ['board_strategy','shrink_volume','sector_rotation','ma_breakout_wrapper',
                'first_board_strategy','dragon_rebound_strategy','broken_board_rebound_strategy','ml_strategy']:
        __import__(f'strategy.{mod}')
    from strategy import StrategyRegistry
    from backtest.vectorized_engine import VectorizedBacktestEngine
    from ml.predictor import MLPredictor
    e = VectorizedBacktestEngine(initial_capital=5000, start_date='2024-10-08',
                                  end_date='2024-12-31', max_hold_stocks=5, single_stock_position=0.15)
    e.load_data()
    e.constraint_matrix = e.build_constraint_matrix()
    p = MLPredictor(); p.load('20260507')
    rows = []
    for name in StrategyRegistry.list_all():
        s = StrategyRegistry.get(name)
        if 'ML' in name and p.model: s._predictor = p
        r = e._run_with_data(s)
        rows.append({'strategy': name, 'return_pct': f"{r.total_return*100:.2f}%",
                     'sharpe': f"{r.sharpe_ratio:.2f}", 'trades': r.total_trades,
                     'win_rate': f"{r.win_rate:.0f}%"})
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv('daily_backtest_results.csv', index=False)

    # 3. Stock pick
    print("\n[3/4] Running stock picker...")
    from stock_picker.picker_v2 import DailyStockPickerV2
    picker = DailyStockPickerV2(['打板策略', '断板反包策略', '均线突破策略'])
    df_data, trade_date = picker.get_latest_data()
    if df_data is not None and not df_data.empty:
        picks = picker.pick(df_data, trade_date)
        if picks is not None and not picks.empty:
            print(picks[['ts_code','name','combined_score','strategy_count']].head(10).to_string())
        else:
            print("  No picks today")
    else:
        print("  No data available")

    # 4. Report
    print("\n[4/4] Generating report...")
    from backtest.report import BacktestReport
    s_best = StrategyRegistry.get('打板策略')
    r_best = e._run_with_data(s_best)
    report = BacktestReport(r_best, '打板策略', output_dir='.')
    path = report.generate()
    print(f"  Report: {path}")

    elapsed = time.time() - t0
    print(f"\nPipeline complete in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
