"""
龙头反包策略 — 强势股回调后的二次介入

核心逻辑：
1. 识别龙头股（前3日涨幅>20%+市场辨识度）
2. 等待回调（回踩5日线/10日线/20日线）
3. 缩量企稳后反包信号介入

反包形态：低开高走，收盘价>前日最高价
"""
import pandas as pd
import numpy as np
import re
from .base import StrategyBase, StrategyRegistry


@StrategyRegistry.register
class DragonReboundStrategy(StrategyBase):
    name = "龙头反包策略"
    version = "1.0"

    def filter(self, df):
        if df.empty:
            return df
        df = df.copy()
        if 'name' in df.columns:
            df = df[~df['name'].str.contains('ST|退市', na=False)]
        if df.empty:
            return df
        if 'close' not in df.columns:
            return pd.DataFrame()
        df = df.sort_values(['ts_code', 'trade_date'])
        g = df.groupby('ts_code')
        df['_ret_3d'] = g['close'].transform(lambda x: x.pct_change(3))
        df['_ret_5d'] = g['close'].transform(lambda x: x.pct_change(5))
        df['_ma5'] = g['close'].transform(lambda x: x.rolling(5).mean())
        df['_ma10'] = g['close'].transform(lambda x: x.rolling(10).mean())
        df['_ma20'] = g['close'].transform(lambda x: x.rolling(20).mean())
        df['_high_prev'] = g['high'].shift(1)
        df['_vol_5d_avg'] = g['vol'].transform(lambda x: x.rolling(5).mean().shift(1))
        df['_prev_close'] = g['close'].shift(1)
        # 龙头条件：近3日或5日涨幅显著
        df = df[(df['_ret_3d'] > 0.15) | (df['_ret_5d'] > 0.25)]
        if df.empty:
            return df
        # 回调到均线附近
        near_ma5 = (df['close'] - df['_ma5']).abs() / df['_ma5'].replace(0, np.nan) <= 0.03
        near_ma10 = (df['close'] - df['_ma10']).abs() / df['_ma10'].replace(0, np.nan) <= 0.03
        near_ma20 = (df['close'] - df['_ma20']).abs() / df['_ma20'].replace(0, np.nan) <= 0.03
        df = df[near_ma5 | near_ma10 | near_ma20]
        if df.empty:
            return df
        # 缩量
        df = df[df['vol'] < df['_vol_5d_avg'] * 0.8]
        # 反包信号：收盘高于前日最高，且开盘低于前日收盘(低开高走)
        df['_is_rebound'] = (df['close'] > df['_high_prev']) & (df['open'] < df['_prev_close'])
        df = df[df['_is_rebound']]
        if df.empty:
            return df
        if 'float_market_cap' in df.columns:
            df = df[(df['float_market_cap'] >= 20)]
        return df.reset_index(drop=True)

    def score(self, df):
        if df.empty:
            return df
        df = df.copy()
        df['total_score'] = 0.0
        for item_name, (score, condition, _) in self._get_rules().items():
            try:
                fields = re.findall(r'[a-zA-Z_]+', condition)
                if not all(f in df.columns for f in fields):
                    continue
                mask = df.eval(condition)
                df.loc[mask, 'total_score'] += score
            except Exception:
                continue
        pass_score = 6
        actual_max = df['total_score'].max()
        effective_pass = max(3, min(pass_score, actual_max * 0.3)) if actual_max > 0 else pass_score
        df = df[df['total_score'] >= effective_pass]
        return df.sort_values('total_score', ascending=False).reset_index(drop=True)

    def generate_signals_vectorized(self, df):
        if df.empty or 'trade_date' not in df.columns:
            return pd.DataFrame()
        try:
            scored = self.score(self.filter(df))
            if scored.empty or 'total_score' not in scored.columns:
                return pd.DataFrame()
            scored['signal'] = (scored['total_score'] >= 8).astype(int)
            return scored.pivot_table(
                index='trade_date', columns='ts_code', values='signal', fill_value=0
            ).astype(int)
        except Exception:
            return pd.DataFrame()

    def _get_rules(self):
        return {
            '回踩5日线': (3, "_ma5 > 0 and abs(close - _ma5) / _ma5 <= 0.02", {}),
            '回踩10日线': (2, "_ma10 > 0 and abs(close - _ma10) / _ma10 <= 0.02", {}),
            '缩量明显': (3, "vol < _vol_5d_avg * 0.6", {}),
            '反包力度强': (3, "close > _high_prev * 1.01", {}),
            '20日涨幅排序': (2, "_ret_5d > 0.3", {}),
            '流通市值适中': (1, "float_market_cap <= 200", {}),
            '换手活跃': (2, "turnover_rate >= 5", {}),
        }
