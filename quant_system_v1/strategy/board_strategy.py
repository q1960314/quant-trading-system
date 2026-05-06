"""
打板策略 — 23维评分追涨停
"""
import pandas as pd
import numpy as np
import re
from .base import StrategyBase, StrategyRegistry


@StrategyRegistry.register
class LimitUpStrategy(StrategyBase):
    name = "打板策略"
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

        # ST过滤
        if 'name' in df.columns:
            df = df[~df['name'].str.contains('ST|退市', na=False)]

        # 涨停筛选：优先用 limit_status，备选用 pct_chg
        has_limit = ('limit_status' in df.columns and (df['limit_status'] == 'U').sum() > 0)
        if has_limit:
            df = df[df['limit_status'] == 'U']
        elif 'pct_chg' in df.columns:
            df = df[df['pct_chg'] >= 9.5]

        if df.empty: return df

        # 有价格
        if 'close' in df.columns:
            df = df[df['close'] > 0]

        # 封单比 — 仅当有真实封单数据时才过滤
        bc = self.sc
        if 'order_amount' in df.columns and 'float_market_cap' in df.columns:
            # 检测是否为真实stk_limit数据（非默认值）
            unique_oa = df['order_amount'].nunique()
            if unique_oa > 5:  # 真实数据应有多种封单金额
                df['order_ratio'] = df['order_amount'] / df['float_market_cap'].replace(0, 1) / 10000
                df = df[df['order_ratio'] >= bc.get('min_order_ratio', 0.03)]

        # 炸板次数 — 仅当有真实数据
        if 'break_limit_times' in df.columns and df['break_limit_times'].nunique() > 1:
            df = df[df['break_limit_times'] <= bc.get('max_break_times', 1)]

        # 连板范围 — 仅当有真实数据
        link_range = bc.get('link_board_range', [2, 4])
        if 'up_down_times' in df.columns and df['up_down_times'].nunique() > 1:
            df = df[(df['up_down_times'] >= link_range[0]) & (df['up_down_times'] <= link_range[1])]

        # 排除尾盘封板 — 仅当有真实数据
        if bc.get('exclude_late_board', True) and 'first_limit_time' in df.columns:
            times = df['first_limit_time'].astype(str)
            if (times != '10:00').sum() > 0:  # 有非默认值
                df = df[times <= '14:40']

        return df.reset_index(drop=True)

    def score(self, df):
        if df.empty: return df
        df = df.copy()
        df['total_score'] = 0.0
        max_possible = 0
        active_dims = 0

        from utils.logger import get_logger
        logger = get_logger("strategy")

        for item_name, (score, condition, weight_dict) in self.sr['items'].items():
            weight = weight_dict.get(self.name, 1)
            if weight == 0:
                continue
            try:
                fields = re.findall(r'[a-zA-Z_]+', condition)
                if not all(f in df.columns for f in fields):
                    continue
                mask = df.eval(condition)
                max_possible += score * weight
                count = mask.sum()
                if count > 0:
                    df.loc[mask, 'total_score'] += score * weight
                    active_dims += 1
            except Exception:
                continue

        # 动态及格线：实际最高分的50%，最少6分
        actual_max = df['total_score'].max()
        effective_pass = max(6, min(self.pass_score, actual_max * 0.55))
        logger.info(f"  评分: {active_dims}维 | 理论={max_possible} | 实际最高={actual_max:.0f} | 及格={effective_pass:.0f}")

        df = df[df['total_score'] >= effective_pass]
        return df.sort_values('total_score', ascending=False).reset_index(drop=True)

    def generate_signals_vectorized(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """向量化信号生成：对所有日期+股票批量计算 buy 信号矩阵 (T,N)。"""
        if df.empty or 'trade_date' not in df.columns or 'ts_code' not in df.columns:
            return pd.DataFrame()
        try:
            filtered = self.filter(df)
            if filtered.empty:
                return pd.DataFrame()
            scored = self.score(filtered)
            if scored.empty or 'total_score' not in scored.columns:
                return pd.DataFrame()
            actual_max = scored['total_score'].max()
            effective_pass = max(6, min(self.pass_score, actual_max * 0.55))
            scored['signal'] = (scored['total_score'] >= effective_pass).astype(int)
            signal_matrix = scored.pivot_table(
                index='trade_date', columns='ts_code', values='signal', fill_value=0
            )
            return signal_matrix.astype(int)
        except Exception:
            return pd.DataFrame()
