"""
Factor IC/ICIR analysis for all registered factors.

Usage: python scripts/run_factor_ic.py
"""
import os, sys, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

from backtest.vectorized_engine import VectorizedBacktestEngine
from factor_lib.registry import FactorRegistry
from factor_lib.evaluation import FactorEvaluator


def main():
    # Load 2024 Q4 data for factor evaluation
    print("Loading data...", flush=True)
    engine = VectorizedBacktestEngine(start_date="2024-10-08", end_date="2024-12-31")
    engine.load_data()
    df = engine.df
    print(f"Data: {len(df)} rows, {df['ts_code'].nunique()} stocks, {df['trade_date'].nunique()} days")

    # Compute all factors
    print("Computing factors...", flush=True)
    factor_names = FactorRegistry.list_all()
    factor_df = FactorRegistry.compute(df, factor_names)
    print(f"Computed {len(factor_names)} factors")

    # Forward returns
    df['_fwd_1d'] = df.groupby('ts_code')['close'].transform(lambda x: x.pct_change().shift(-1))
    df['_fwd_5d'] = df.groupby('ts_code')['close'].transform(lambda x: x.pct_change(5).shift(-5))

    # Evaluate each factor
    print("Evaluating IC/ICIR...", flush=True)
    evaluator = FactorEvaluator(min_obs=50, icir_threshold=0.0)
    results = []
    for name in factor_names:
        if name not in factor_df.columns:
            continue
        vals = factor_df[name]
        rets = df['_fwd_1d']
        ic_info = evaluator.ic_analysis(vals, rets)
        results.append({
            'factor': name,
            'IC_mean': ic_info['ic_mean'],
            'ICIR': ic_info['icir'],
            'IC_win_rate': ic_info['ic_win_rate'],
        })

    results_df = pd.DataFrame(results).sort_values('ICIR', key=abs, ascending=False)

    print(f"\n{'='*70}")
    print(f"  Factor IC Analysis — Top 30 by |ICIR|")
    print(f"{'='*70}")
    print(f"{'Factor':<30} {'IC Mean':>8} {'ICIR':>8} {'Win Rate':>8}")
    print(f"{'-'*70}")
    for _, row in results_df.head(30).iterrows():
        print(f"{row['factor']:<30} {row['IC_mean']:>8.4f} {row['ICIR']:>8.3f} {row['IC_win_rate']:>7.1%}")
    print(f"{'='*70}")

    # Summary stats
    passed = (results_df['ICIR'].abs() >= 0.3).sum()
    top_5 = results_df.head(5)['factor'].tolist()
    print(f"Passed (|ICIR| >= 0.3): {passed}/{len(results_df)}")
    print(f"Top 5 factors: {top_5}")

    # Save
    out = "factor_ic_analysis.csv"
    results_df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return results_df


if __name__ == '__main__':
    main()
