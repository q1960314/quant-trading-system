"""
均线突破策略 — StrategyRegistry 适配器
将 strategies/ma_breakout_strategy.py 桥接到统一策略管线
"""
import pandas as pd
import numpy as np
import sys, os

# 确保 strategies 目录可导入
_strategies_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'strategies')
if _strategies_dir not in sys.path:
    sys.path.insert(0, _strategies_dir)

from .base import StrategyBase, StrategyRegistry


@StrategyRegistry.register
class MABreakoutWrapper(StrategyBase):
    name = "均线突破策略"
    version = "1.0"

    def __init__(self, config=None):
        super().__init__(config)
        from ma_breakout_strategy import MABreakoutStrategy
        self._engine = MABreakoutStrategy()

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        if 'name' in df.columns:
            df = df[~df['name'].str.contains('ST|退市', na=False)]
        if 'vol' in df.columns:
            df = df[df['vol'] > 0]
        return df.reset_index(drop=True)

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df['total_score'] = 0.0
        # 使用内嵌引擎生成信号并转换为分数
        try:
            for code in df['ts_code'].unique():
                code_df = df[df['ts_code'] == code].sort_values('trade_date')
                if len(code_df) < 20:
                    continue
                signals = self._engine.run(code_df.copy())
                if signals:
                    # 最后一条信号转为分数
                    last = signals[-1]
                    if hasattr(last, 'signal_type'):
                        stype = last.signal_type.value if hasattr(last.signal_type, 'value') else str(last.signal_type)
                        if stype in ('golden_cross', 'buy'):
                            df.loc[code_df.index[-1], 'total_score'] = 15
                        elif stype in ('death_cross',):
                            df.loc[code_df.index[-1], 'total_score'] = -5
        except Exception:
            pass
        return df

    def generate_signals_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or 'trade_date' not in df.columns or 'ts_code' not in df.columns:
            return pd.DataFrame()
        try:
            scored = self.score(df)
            if scored.empty or 'total_score' not in scored.columns:
                return pd.DataFrame()
            scored['signal'] = (scored['total_score'] >= 10).astype(int)
            return scored.pivot_table(
                index='trade_date', columns='ts_code', values='signal', fill_value=0
            ).astype(int)
        except Exception:
            return pd.DataFrame()
