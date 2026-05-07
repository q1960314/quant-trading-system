"""
龙头反包策略 — 强势股回调反包 (simplified v3)
"""
import pandas as pd; import numpy as np
from .base import StrategyBase, StrategyRegistry


@StrategyRegistry.register
class DragonReboundStrategy(StrategyBase):
    name = "龙头反包策略"
    version = "3.0"

    def filter(self, df):
        if df.empty: return df
        df = df.copy()
        if 'name' in df.columns: df = df[~df['name'].str.contains('ST|退市', na=False)]
        if 'close' not in df.columns or 'high' not in df.columns: return pd.DataFrame()
        if 'amount' in df.columns: df = df[df['amount'] >= 50000]
        # Recent strong performance: pct_chg > 3% (daily leader)
        df = df.sort_values(['ts_code', 'trade_date'])
        g = df.groupby('ts_code')
        df['_prev_high'] = g['high'].shift(1)
        df['_prev_close'] = g['close'].shift(1)
        # Rebound: today's close > yesterday's high, open < yesterday's close
        df['_is_rebound'] = (df['close'] > df['_prev_high']) & (df['open'] < df['_prev_close'])
        df = df[df['_is_rebound']]
        # Recent positive trend
        df['_ret_3d'] = g['close'].transform(lambda x: x.pct_change(3))
        df = df[df['_ret_3d'] > 0.05]  # Relaxed from 0.10
        if 'float_market_cap' in df.columns:
            df = df[(df['float_market_cap'] >= 10)]
        return df.reset_index(drop=True)

    def score(self, df):
        if df.empty: return df
        df = df.copy()
        df['total_score'] = 3  # Base for engulfing pattern
        if '_ret_3d' in df.columns:
            df.loc[df['_ret_3d'] > 0.15, 'total_score'] += 5
            df.loc[(df['_ret_3d'] > 0.08) & (df['_ret_3d'] <= 0.15), 'total_score'] += 3
        if 'pct_chg' in df.columns:
            df.loc[df['pct_chg'] > 5, 'total_score'] += 4
            df.loc[(df['pct_chg'] > 2) & (df['pct_chg'] <= 5), 'total_score'] += 2
        if 'turnover_rate' in df.columns:
            df.loc[(df['turnover_rate'] >= 5) & (df['turnover_rate'] <= 35), 'total_score'] += 2
        if 'float_market_cap' in df.columns:
            df.loc[(df['float_market_cap'] >= 20) & (df['float_market_cap'] <= 300), 'total_score'] += 1
        pass_score = 5
        df = df[df['total_score'] >= pass_score]
        return df.sort_values('total_score', ascending=False).reset_index(drop=True)

    def generate_signals_vectorized(self, df):
        if df.empty or 'trade_date' not in df.columns: return pd.DataFrame()
        try:
            scored = self.score(self.filter(df))
            if scored.empty or 'total_score' not in scored.columns: return pd.DataFrame()
            scored['signal'] = (scored['total_score'] >= 6).astype(int)
            return scored.pivot_table(index='trade_date', columns='ts_code', values='signal', fill_value=0).astype(int)
        except Exception: return pd.DataFrame()
