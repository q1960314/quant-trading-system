"""
断板反包策略 — 涨停断板后的修复行情

核心逻辑：
1. 识别涨停断板（触及涨停但未能封住，或炸板回落）
2. 次日低开不破关键支撑（前日涨停价/5日线）
3. 第三日反包收复断板失地

断板特征：
- 当日最高触及涨停但收盘远离涨停价
- 成交量大于前日2倍以上（多空分歧激烈）
- 次日低开但不破位
"""
import pandas as pd
import numpy as np
import re
from .base import StrategyBase, StrategyRegistry


@StrategyRegistry.register
class BrokenBoardReboundStrategy(StrategyBase):
    name = "断板反包策略"
    version = "1.0"

    def filter(self, df):
        if df.empty:
            return df
        df = df.copy()
        if 'name' in df.columns:
            df = df[~df['name'].str.contains('ST|退市', na=False)]
        if df.empty:
            return df
        if 'high' not in df.columns or 'close' not in df.columns:
            return pd.DataFrame()
        df = df.sort_values(['ts_code', 'trade_date'])
        g = df.groupby('ts_code')
        df['_pre_close'] = g['close'].shift(1)
        df['_limit_price'] = df['_pre_close'] * 1.10
        df['_hit_limit'] = df['high'] >= df['_limit_price']
        df['_close_gap'] = (df['_limit_price'] - df['close']) / df['_limit_price'].replace(0, np.nan)
        df['_prev_vol'] = g['vol'].shift(1)
        df['_vol_ratio'] = df['vol'] / df['_prev_vol'].replace(0, np.nan)
        df['_prev_high'] = g['high'].shift(1)
        df['_open_gap'] = (df['open'] - df['_pre_close']) / df['_pre_close'].replace(0, np.nan)
        # 前日是断板日
        df['_broken_prev'] = (
            g['_hit_limit'].shift(1) &
            (g['_close_gap'].shift(1) > 0.02) &
            (g['_vol_ratio'].shift(1) > 1.5)
        )
        df = df[df['_broken_prev']]
        if df.empty:
            return df
        # 当前日低开但不破位
        df = df[df['_open_gap'] > -0.05]
        # 缩量
        df = df[df['_vol_ratio'] < 0.8]
        # 当前日反包：收盘>前日收盘
        df['_prev_close'] = g['close'].shift(1)
        df['_rebound'] = df['close'] > df['_prev_close']
        df = df[df['_rebound']]
        if df.empty:
            return df
        if 'float_market_cap' in df.columns:
            df = df[(df['float_market_cap'] >= 10)]
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
        pass_score = 8
        actual_max = df['total_score'].max()
        effective_pass = max(5, min(pass_score, actual_max * 0.5)) if actual_max > 0 else pass_score
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
            '断板幅度小': (4, "_close_gap <= 0.05", {}),
            '缩量止跌': (3, "_vol_ratio <= 0.6", {}),
            '反包力度强': (3, "close > _prev_close * 1.02", {}),
            '低开幅度小': (2, "_open_gap >= -0.03", {}),
            '流通盘适中': (2, "float_market_cap >= 20 and float_market_cap <= 300", {}),
            '前日放量充分': (1, "_vol_ratio >= 2", {}),
        }
