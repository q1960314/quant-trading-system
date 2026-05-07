"""
Factor IC/ICIR analysis — cross-sectional by trade_date.

Usage: python scripts/run_factor_ic.py [--top 30]
"""
import os, sys, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

from backtest.vectorized_engine import VectorizedBacktestEngine
from factor_lib.registry import FactorRegistry


def main():
    print("Loading data...", flush=True)
    engine = VectorizedBacktestEngine(start_date="2024-10-08", end_date="2024-12-31")
    engine.load_data()
    df = engine.df
    print(f"Data: {len(df)} rows, {df['ts_code'].nunique()} stocks, {df['trade_date'].nunique()} days")

    factor_names = FactorRegistry.list_all()
    print(f"Computing {len(factor_names)} factors...", flush=True)
    factor_df = FactorRegistry.compute(df, factor_names)

    df['_fwd_1d'] = df.groupby('ts_code')['close'].transform(lambda x: x.pct_change().shift(-1))

    print("Computing cross-sectional IC by date...", flush=True)
    dates = sorted(df['trade_date'].unique())
    results = []

    for fname in factor_names:
        if fname not in factor_df.columns:
            continue
        ic_list = []
        for d in dates:
            mask = (df['trade_date'] == d)
            fv = factor_df.loc[mask, fname]
            fr = df.loc[mask, '_fwd_1d']
            valid = fv.notna() & fr.notna()
            if valid.sum() < 30:
                continue
            ic = fv[valid].corr(fr[valid])
            if not np.isnan(ic):
                ic_list.append(ic)
        if len(ic_list) < 10:
            continue
        ic_mean = np.mean(ic_list)
        ic_std = np.std(ic_list)
        icir = ic_mean / ic_std if ic_std > 0 else 0
        results.append({
            'factor': fname, 'IC_mean': ic_mean, 'IC_std': ic_std,
            'ICIR': icir, 'IC_win_rate': sum(1 for x in ic_list if x > 0) / len(ic_list),
            'obs': len(ic_list),
        })

    results_df = pd.DataFrame(results).sort_values('ICIR', key=abs, ascending=False)

    print(f"\n{'='*70}")
    print(f"  Factor IC Analysis — Top 30 by |ICIR|")
    print(f"{'='*70}")
    print(f"{'Factor':<28} {'IC Mean':>8} {'ICIR':>8} {'WinRate':>7} {'Obs':>6}")
    print(f"{'-'*70}")
    for _, row in results_df.head(30).iterrows():
        print(f"{row['factor']:<28} {row['IC_mean']:>8.4f} {row['ICIR']:>8.3f} {row['IC_win_rate']:>6.1%} {row['obs']:>5}")
    print(f"{'='*70}")

    passed = (results_df['ICIR'].abs() >= 0.3).sum()
    top5 = results_df.head(5)['factor'].tolist()
    print(f"Passed |ICIR|>=0.3: {passed}/{len(results_df)}")
    print(f"Top 5: {top5}")

    results_df.to_csv("factor_ic_analysis_v2.csv", index=False)
    print("Saved: factor_ic_analysis_v2.csv")


if __name__ == '__main__':
    main()
