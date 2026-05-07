"""
板块轮动策略 — 抓行业资金流向 (simplified v2)
"""
import pandas as pd; import numpy as np
from .base import StrategyBase, StrategyRegistry


@StrategyRegistry.register
class SectorRotationStrategy(StrategyBase):
    name = "板块轮动策略"
    version = "2.1"

    def filter(self, df):
        if df.empty: return df
        df = df.copy()
        if 'name' in df.columns: df = df[~df['name'].str.contains('ST|退市', na=False)]
        if 'close' in df.columns: df = df[df['close'] > 0]
        if 'amount' in df.columns: df = df[df['amount'] >= 50000]
        # Sector leader: top stocks by price change within industry
        if 'industry' in df.columns and 'pct_chg' in df.columns:
            df['_ind_rank'] = df.groupby(['trade_date', 'industry'])['pct_chg'].transform(lambda x: x.rank(ascending=False, pct=True))
            df = df[df['_ind_rank'] <= 0.1]  # Top 10% in industry (tightened from 30%)
        return df.reset_index(drop=True)

    def score(self, df):
        if df.empty: return df
        df = df.copy()
        df['total_score'] = 0.0
        if 'pct_chg' in df.columns:
            df.loc[df['pct_chg'] > 3, 'total_score'] += 5
            df.loc[(df['pct_chg'] > 1) & (df['pct_chg'] <= 3), 'total_score'] += 3
            df.loc[(df['pct_chg'] > 0) & (df['pct_chg'] <= 1), 'total_score'] += 1
        if 'turnover_rate' in df.columns:
            df.loc[(df['turnover_rate'] >= 3) & (df['turnover_rate'] <= 20), 'total_score'] += 3
        if 'net_amount' in df.columns:
            df.loc[df['net_amount'] > 0, 'total_score'] += 2
        if 'float_market_cap' in df.columns:
            df.loc[(df['float_market_cap'] >= 20) & (df['float_market_cap'] <= 500), 'total_score'] += 1
        if '_ind_rank' in df.columns:
            df.loc[df['_ind_rank'] <= 0.1, 'total_score'] += 2
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
