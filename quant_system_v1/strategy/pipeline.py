"""
策略管线 v2 — 统一 filter→factor→score→rank 流程 + 市场状态检测
所有策略复用此管线，差异仅在于配置不同
"""
import pandas as pd
import numpy as np
import re
from abc import ABC, abstractmethod
from utils.logger import get_logger
from factor_lib import FactorRegistry

logger = get_logger("pipeline")


class MarketRegimeDetector:
    """市场状态检测器：综合指数趋势+波动率+涨停情绪+北向资金判定牛/熊/震荡"""

    def __init__(self, index_data=None, macro_data=None):
        self.index_data = index_data or pd.DataFrame()
        self.macro_data = macro_data or pd.DataFrame()
        self.current_regime = "normal"  # bull / bear / normal

    def detect(self, df_today):
        """输入当日全市场数据，返回市场状态和强度分数"""
        if df_today is None or df_today.empty:
            return {"regime": "normal", "score": 0.5, "confidence": 0.3}

        signals = {}

        # 1. 涨停家数占比
        if 'limit_status' in df_today.columns:
            total = len(df_today)
            limit_up = (df_today['limit_status'] == 'U').sum()
            ratio = limit_up / total if total > 0 else 0
            if ratio > 0.05: signals['limit_ratio'] = 1   # 极多涨停 → 牛市特征
            elif ratio > 0.02: signals['limit_ratio'] = 0.5
            else: signals['limit_ratio'] = -0.5

        # 2. 涨跌比
        if 'pct_chg' in df_today.columns:
            up = (df_today['pct_chg'] > 0).sum()
            down = (df_today['pct_chg'] < 0).sum()
            if down > 0:
                udr = up / down
                if udr > 3: signals['ud_ratio'] = 1
                elif udr > 1.5: signals['ud_ratio'] = 0.5
                elif udr < 0.5: signals['ud_ratio'] = -1
                else: signals['ud_ratio'] = 0

        # 3. 北向资金净流入
        if 'net_amount' in df_today.columns:
            net = df_today['net_amount'].sum()
            if net > 1e8: signals['north'] = 1
            elif net > 0: signals['north'] = 0.5
            else: signals['north'] = -0.5

        # 4. 综合判断
        score = sum(signals.values()) / max(len(signals), 1) if signals else 0
        if score > 0.3: regime = "bull"
        elif score < -0.2: regime = "bear"
        else: regime = "normal"

        self.current_regime = regime
        return {"regime": regime, "score": score, "confidence": abs(score), "signals": signals}


class Pipeline:
    """策略管线：filter → factor_compute → score → rank"""

    def __init__(self, strategy_name, filter_fn, score_rules, factor_list=None,
                 risk_params=None, pass_score=10):
        self.name = strategy_name
        self._filter_fn = filter_fn
        self._score_rules = score_rules  # {item_name: [score, condition, weights]}
        self._factor_list = factor_list or []
        self._risk = risk_params or {}
        self.pass_score = pass_score
        self.regime_detector = MarketRegimeDetector()

    def run(self, df, feed_regime=True):
        """完整管线"""
        if df.empty: return df

        # 1. 过滤
        df = self._filter_fn(df.copy())
        if df.empty: return df

        # 2. 因子计算
        if self._factor_list:
            factor_df = FactorRegistry.compute(df, self._factor_list)
            for col in factor_df.columns:
                if col not in df.columns:
                    df[col] = factor_df[col]

        # 3. 市场状态感知
        if feed_regime:
            regime_info = self.regime_detector.detect(df)
            self._regime = regime_info['regime']
            logger.debug(f"  市场状态: {regime_info['regime']} (score={regime_info['score']:.2f})")

        # 4. 评分
        df['total_score'] = 0.0
        for item_name, (score, condition, weights) in self._score_rules.items():
            w = weights.get(self.name, 1)
            if w == 0: continue
            try:
                fields = re.findall(r'[a-zA-Z_]+', condition)
                if not all(f in df.columns for f in fields): continue
                mask = df.eval(condition)
                if mask.sum() > 0:
                    # 牛市放大进攻因子，熊市放大防守因子
                    if hasattr(self, '_regime'):
                        if self._regime == "bull" and 'bool' not in condition:
                            w *= 1.2
                        elif self._regime == "bear" and 'stop' in item_name:
                            w *= 1.3
                    df.loc[mask, 'total_score'] += score * w
            except Exception: pass

        # 5. 动态及格线
        actual_max = df['total_score'].max()
        effective_pass = max(6, min(self.pass_score, actual_max * 0.5)) if actual_max > 0 else self.pass_score

        df = df[df['total_score'] >= effective_pass]
        return df.sort_values('total_score', ascending=False).reset_index(drop=True)
