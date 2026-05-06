"""
首板策略 — 专注首板涨停次日溢价

核心逻辑：
1. 识别首板（近20日首次涨停）
2. 评估封板质量（封板时间/封单量/炸板次数）
3. 次日开盘买入，博弈1-3天溢价

首板优势：抛压轻、上方无套牢盘、辨识度高
"""
import pandas as pd
import numpy as np
import re
from .base import StrategyBase, StrategyRegistry


@StrategyRegistry.register
class FirstBoardStrategy(StrategyBase):
    name = "首板策略"
    version = "1.0"

    def __init__(self, config=None):
        super().__init__(config)
        self._risk = None

    def filter(self, df):
        if df.empty:
            return df
        df = df.copy()
        if 'name' in df.columns:
            df = df[~df['name'].str.contains('ST|退市', na=False)]
        if df.empty:
            return df
        has_limit = 'limit_status' in df.columns
        if has_limit:
            df = df.sort_values(['ts_code', 'trade_date'])
            df['_is_limit'] = df['limit_status'].apply(
                lambda x: 1 if str(x).upper() in ('U', '1') else 0
            )
            df['_limit_20d'] = (
                df.groupby('ts_code')['_is_limit']
                .rolling(20, min_periods=1).sum()
                .shift(1).reset_index(0, drop=True)
            )
            df = df[(df['_is_limit'] == 1) & (df['_limit_20d'].fillna(0) <= 1)]
        elif 'pct_chg' in df.columns:
            df = df[df['pct_chg'] >= 9.5]
        if df.empty:
            return df
        if 'close' in df.columns:
            df = df[df['close'] > 0]
        if 'float_market_cap' in df.columns:
            fmc = df['float_market_cap']
            df = df[(fmc >= 10) & (fmc <= 500)]
        if 'turnover_rate' in df.columns:
            df = df[(df['turnover_rate'] >= 1) & (df['turnover_rate'] <= 30)]
        return df.reset_index(drop=True)

    def score(self, df):
        if df.empty:
            return df
        df = df.copy()
        df['total_score'] = 0.0
        for item_name, (score, condition, weight_dict) in self._get_rules().items():
            try:
                fields = re.findall(r'[a-zA-Z_]+', condition)
                if not all(f in df.columns for f in fields):
                    continue
                mask = df.eval(condition)
                df.loc[mask, 'total_score'] += score
            except Exception:
                continue
        pass_score = 4
        actual_max = df['total_score'].max()
        effective_pass = max(2, min(pass_score, actual_max * 0.3)) if actual_max > 0 else pass_score
        df = df[df['total_score'] >= effective_pass]
        return df.sort_values('total_score', ascending=False).reset_index(drop=True)

    def generate_signals_vectorized(self, df):
        if df.empty or 'trade_date' not in df.columns:
            return pd.DataFrame()
        try:
            filtered = self.filter(df)
            if filtered.empty:
                return pd.DataFrame()
            scored = self.score(filtered)
            if scored.empty or 'total_score' not in scored.columns:
                return pd.DataFrame()
            pass_score = 6
            scored['signal'] = (scored['total_score'] >= pass_score).astype(int)
            return scored.pivot_table(
                index='trade_date', columns='ts_code', values='signal', fill_value=0
            ).astype(int)
        except Exception:
            return pd.DataFrame()

    def _get_rules(self):
        return {
            '封板时间早': (4, "first_limit_time <= '10:30'", {}),
            '封单量充足': (3, "order_ratio >= 0.05", {}),
            '未炸板': (3, "break_limit_times == 0", {}),
            '小市值爆发力': (2, "float_market_cap <= 100", {}),
            '换手率适中': (2, "turnover_rate >= 3 and turnover_rate <= 15", {}),
            '量比放大': (2, "volume_ratio >= 2", {}),
            '非尾盘偷袭': (2, "first_limit_time <= '14:00'", {}),
            '均线多头': (2, "ma5_cross_ma20 > 0", {}),
        }
