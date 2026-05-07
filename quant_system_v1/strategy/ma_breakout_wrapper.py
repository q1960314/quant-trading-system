"""
均线突破策略 — MA5上穿MA20 + 放量 (simplified v2)
"""
import pandas as pd; import numpy as np
from .base import StrategyBase, StrategyRegistry


@StrategyRegistry.register
class MABreakoutWrapper(StrategyBase):
    name = "均线突破策略"
    version = "2.1"

    def filter(self, df):
        if df.empty: return df
        df = df.copy()
        if 'name' in df.columns: df = df[~df['name'].str.contains('ST|退市', na=False)]
        if 'close' not in df.columns or 'vol' not in df.columns: return pd.DataFrame()
        df = df.sort_values(['ts_code', 'trade_date'])
        g = df.groupby('ts_code')
        df['_ma5'] = g['close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['_ma20'] = g['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
        df['_prev_ma5'] = g['_ma5'].shift(1)
        df['_prev_ma20'] = g['_ma20'].shift(1)
        df['_vol_ma5'] = g['vol'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        # Golden cross: MA5 crosses above MA20
        df = df[(df['_prev_ma5'] <= df['_prev_ma20']) & (df['_ma5'] > df['_ma20'])]
        # Volume confirmation
        df = df[df['vol'] > df['_vol_ma5'] * 1.2]
        if 'close' in df.columns: df = df[df['close'] > 0]
        if 'amount' in df.columns: df = df[df['amount'] >= 50000]
        return df.reset_index(drop=True)

    def score(self, df):
        if df.empty: return df
        df = df.copy()
        df['total_score'] = 0.0
        df['_cross_strength'] = (df['_ma5'] - df['_ma20']) / df['_ma20'].replace(0, np.nan) * 100
        df.loc[df['_cross_strength'] > 2, 'total_score'] += 5
        df.loc[(df['_cross_strength'] > 1) & (df['_cross_strength'] <= 2), 'total_score'] += 3
        if 'pct_chg' in df.columns:
            df.loc[df['pct_chg'] > 0, 'total_score'] += 2
            df.loc[df['pct_chg'] > 3, 'total_score'] += 2
        if 'turnover_rate' in df.columns:
            df.loc[(df['turnover_rate'] >= 3) & (df['turnover_rate'] <= 25), 'total_score'] += 2
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
