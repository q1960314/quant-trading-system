"""
首板策略 — 首板涨停识别 (simplified v3)
"""
import pandas as pd; import numpy as np
from .base import StrategyBase, StrategyRegistry


@StrategyRegistry.register
class FirstBoardStrategy(StrategyBase):
    name = "首板策略"
    version = "3.0"

    def filter(self, df):
        if df.empty: return df
        df = df.copy()
        if 'name' in df.columns: df = df[~df['name'].str.contains('ST|退市', na=False)]
        if 'close' in df.columns: df = df[df['close'] > 0]
        if 'amount' in df.columns: df = df[df['amount'] >= 50000]
        # Identify limit-up stocks
        if 'limit_status' in df.columns:
            df = df[df['limit_status'] == 'U']
        elif 'pct_chg' in df.columns:
            df = df[df['pct_chg'] >= 9.5]
        if df.empty: return df
        if 'float_market_cap' in df.columns:
            df = df[(df['float_market_cap'] >= 10) & (df['float_market_cap'] <= 500)]
        return df.reset_index(drop=True)

    def score(self, df):
        if df.empty: return df
        df = df.copy()
        df['total_score'] = 3  # Base score for being limit-up
        if 'first_limit_time' in df.columns:
            times = df['first_limit_time'].astype(str)
            df.loc[times <= '10:00', 'total_score'] += 5
            df.loc[(times > '10:00') & (times <= '10:30'), 'total_score'] += 3
        if 'turnover_rate' in df.columns:
            df.loc[(df['turnover_rate'] >= 3) & (df['turnover_rate'] <= 20), 'total_score'] += 3
        if 'float_market_cap' in df.columns:
            df.loc[(df['float_market_cap'] >= 20) & (df['float_market_cap'] <= 200), 'total_score'] += 2
        if 'break_limit_times' in df.columns:
            df.loc[df['break_limit_times'] == 0, 'total_score'] += 3
        if 'up_down_times' in df.columns:
            df.loc[(df['up_down_times'] >= 1) & (df['up_down_times'] <= 3), 'total_score'] += 2
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
