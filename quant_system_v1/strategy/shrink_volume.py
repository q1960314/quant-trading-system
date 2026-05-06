"""
缩量潜伏策略 — 首板后缩量回调低吸
"""
import pandas as pd
import numpy as np
import re
from .base import StrategyBase, StrategyRegistry


@StrategyRegistry.register
class VolumeContractionStrategy(StrategyBase):
    name = "缩量潜伏策略"
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

        # 有量
        if 'vol' in df.columns:
            df = df[df['vol'] > 0]
        if 'amount' in df.columns:
            df = df[df['amount'] >= 100000]  # 成交额≥1亿

        lc = self.sc

        # 标记首板（需多日数据，单日数据跳过此步骤）
        if 'limit_status' in df.columns and 'ts_code' in df.columns and len(df['trade_date'].unique()) > 1:
            df = df.sort_values(['ts_code', 'trade_date'])
            # 涨停标记
            df['is_limit'] = df['limit_status'].apply(lambda x: 1 if str(x).upper() in ('U', '涨停', '1') else 0)
            # 近20日涨停次数
            df['limit_20d'] = df.groupby('ts_code')['is_limit'].rolling(20, min_periods=1).sum().shift(1).reset_index(0, drop=True)
            df['is_first_board'] = (df['is_limit'] == 1) & (df['limit_20d'].fillna(0) == 0)

            # 首板成交量
            df['board_vol'] = np.where(df['is_first_board'], df['vol'], np.nan)
            df['board_vol'] = df.groupby('ts_code')['board_vol'].ffill()
            df['board_high'] = np.where(df['is_first_board'], df['high'], np.nan)
            df['board_high'] = df.groupby('ts_code')['board_high'].ffill()
            df['board_low'] = np.where(df['is_first_board'], df['low'], np.nan)
            df['board_low'] = df.groupby('ts_code')['board_low'].ffill()

            # 首板放量倍数
            df['vol_5d_avg'] = df.groupby('ts_code')['vol'].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            df['board_vol_growth'] = np.where(df['is_first_board'], df['vol'] / df['vol_5d_avg'].replace(0, np.nan), np.nan)

            # 首板后天数
            df['board_id'] = df.groupby('ts_code')['is_first_board'].cumsum()
            df['days_after_board'] = df.groupby(['ts_code', 'board_id']).cumcount()

            # 当前相对首板的缩量比例
            df['current_vol_ratio'] = df['vol'] / df['board_vol'].replace(0, np.nan)

            # 支撑位偏离
            support = lc.get('pullback_support_level', 0.5)
            df['board_support_price'] = df['board_low'] + (df['board_high'] - df['board_low']) * support
            df['price_to_support_ratio'] = (df['close'] - df['board_support_price']) / df['board_support_price'].replace(0, np.nan)

            # 筛选
            shrink_range = lc.get('shrink_volume_ratio', [1/3, 1/2])
            days_range = lc.get('shrink_days_range', [3, 10])
            tol = lc.get('support_tolerance', 0.02)

            df = df[
                (df['days_after_board'] >= days_range[0]) &
                (df['days_after_board'] <= days_range[1]) &
                (df['current_vol_ratio'] >= shrink_range[0]) &
                (df['current_vol_ratio'] <= shrink_range[1]) &
                (df['price_to_support_ratio'].abs() <= tol)
            ]

        return df.reset_index(drop=True)

    def score(self, df):
        if df.empty: return df
        df = df.copy()
        df['total_score'] = 0.0
        df['score_detail'] = ''

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
                df['score_detail'] = df.apply(
                    lambda x, n=item_name, s=score*weight: x['score_detail'] + f"{n}:{s}分;" if mask.loc[x.name] else x['score_detail'],
                    axis=1
                )
            except Exception:
                continue

        actual_max = df['total_score'].max()
        effective_pass = max(3, min(self.pass_score, actual_max * 0.4)) if actual_max > 0 else self.pass_score
        df = df[df['total_score'] >= effective_pass]
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
