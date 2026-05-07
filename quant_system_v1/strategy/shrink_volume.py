"""
缩量潜伏策略 — 首板后缩量回调低吸 (simplified v2)
"""
import pandas as pd; import numpy as np
from .base import StrategyBase, StrategyRegistry


@StrategyRegistry.register
class VolumeContractionStrategy(StrategyBase):
    name = "缩量潜伏策略"
    version = "2.1"

    def filter(self, df):
        if df.empty: return df
        df = df.copy()
        if 'name' in df.columns:
            df = df[~df['name'].str.contains('ST|退市', na=False)]
        # Basic: positive price, minimum liquidity
        if 'close' in df.columns: df = df[df['close'] > 0]
        if 'amount' in df.columns: df = df[df['amount'] >= 50000]
        if 'vol' not in df.columns: return pd.DataFrame()
        # Volume shrinking: today's volume < 50% of 5-day average
        df = df.sort_values(['ts_code', 'trade_date'])
        g = df.groupby('ts_code')
        df['_vol_ma5'] = g['vol'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
        df['_vol_ratio'] = df['vol'] / df['_vol_ma5'].replace(0, np.nan)
        df = df[df['_vol_ratio'] < 0.5]
        # Price near 10-day MA (pullback)
        df['_ma10'] = g['close'].transform(lambda x: x.rolling(10, min_periods=1).mean())
        df['_dev_pct'] = abs(df['close'] - df['_ma10']) / df['_ma10'].replace(0, np.nan)
        df = df[(df['_dev_pct'] <= 0.03) & (df['close'] < df['_ma10'])]
        if 'pct_chg' in df.columns: df = df[df['pct_chg'] > -5]  # Not crashing
        return df.reset_index(drop=True)

    def score(self, df):
        if df.empty: return df
        df = df.copy()
        df['total_score'] = 0.0
        if '_vol_ratio' in df.columns:
            df.loc[df['_vol_ratio'] < 0.3, 'total_score'] += 4
            df.loc[(df['_vol_ratio'] >= 0.3) & (df['_vol_ratio'] < 0.5), 'total_score'] += 2
        if '_dev_pct' in df.columns:
            df.loc[df['_dev_pct'] <= 0.01, 'total_score'] += 3
        if 'pct_chg' in df.columns:
            df.loc[df['pct_chg'] > 0, 'total_score'] += 2
        if 'turnover_rate' in df.columns:
            df.loc[(df['turnover_rate'] >= 2) & (df['turnover_rate'] <= 15), 'total_score'] += 2
        if 'float_market_cap' in df.columns:
            df.loc[(df['float_market_cap'] >= 10) & (df['float_market_cap'] <= 200), 'total_score'] += 1
        pass_score = 4
        df = df[df['total_score'] >= pass_score]
        return df.sort_values('total_score', ascending=False).reset_index(drop=True)

    def generate_signals_vectorized(self, df):
        if df.empty or 'trade_date' not in df.columns: return pd.DataFrame()
        try:
            scored = self.score(self.filter(df))
            if scored.empty or 'total_score' not in scored.columns: return pd.DataFrame()
            scored['signal'] = (scored['total_score'] >= 5).astype(int)
            return scored.pivot_table(index='trade_date', columns='ts_code', values='signal', fill_value=0).astype(int)
        except Exception: return pd.DataFrame()
