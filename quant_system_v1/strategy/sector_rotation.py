"""
板块轮动策略 — 抓行业资金流向轮动
"""
import pandas as pd
import numpy as np
import re
from .base import StrategyBase, StrategyRegistry


@StrategyRegistry.register
class SectorRotationStrategy(StrategyBase):
    name = "板块轮动策略"
    version = "2.0"

    def __init__(self, config=None):
        super().__init__(config)
        from config.strategy_config import STRATEGY_CONFIG, SCORING_RULES
        self.sc = STRATEGY_CONFIG.get(self.name, {})
        self.sr = SCORING_RULES
        self.pass_score = self.sr['strategy_pass_score'].get(self.name, self.sr['pass_score_default'])

    def filter(self, df):
        if df.empty: return df
        df = df.copy()

        if 'name' in df.columns:
            df = df[~df['name'].str.contains('ST|退市', na=False)]

        # 主线行业
        rc = self.sc
        if rc.get('main_trend', True) and 'is_main_industry' in df.columns:
            df = df[df['is_main_industry'] == 1]

        # 资金流入排名
        if rc.get('fund_inflow_top') and 'industry_fund_rank' in df.columns:
            df = df[df['industry_fund_rank'] <= rc['fund_inflow_top']]

        return df.reset_index(drop=True)

    def score(self, df):
        if df.empty: return df
        df = df.copy()
        df['total_score'] = 0.0

        for item_name, (score, condition, weight_dict) in self.sr['items'].items():
            weight = weight_dict.get(self.name, 1)
            if weight == 0:
                continue
            try:
                fields = re.findall(r'[a-zA-Z_]+', condition)
                if not all(f in df.columns for f in fields):
                    continue
                mask = df.eval(condition)
                df.loc[mask, 'total_score'] += score * weight
            except Exception:
                continue

        df = df[df['total_score'] >= self.pass_score]
        return df.sort_values('total_score', ascending=False).reset_index(drop=True)

    def generate_signals_vectorized(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        if df.empty or 'trade_date' not in df.columns or 'ts_code' not in df.columns:
            return pd.DataFrame()
        try:
            filtered = self.filter(df)
            if filtered.empty:
                return pd.DataFrame()
            scored = self.score(filtered)
            if scored.empty or 'total_score' not in scored.columns:
                return pd.DataFrame()
            scored['signal'] = (scored['total_score'] >= self.pass_score).astype(int)
            return scored.pivot_table(
                index='trade_date', columns='ts_code', values='signal', fill_value=0
            ).astype(int)
        except Exception:
            return pd.DataFrame()
