"""ML预测策略 — LightGBM涨跌概率作为信号"""
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base import StrategyBase, StrategyRegistry


@StrategyRegistry.register
class MLPredictionStrategy(StrategyBase):
    name = "ML预测策略"
    version = "1.0"

    def __init__(self, config=None):
        super().__init__(config)
        self._predictor = None

    def _get_predictor(self):
        if self._predictor is None:
            from ml.predictor import MLPredictor
            self._predictor = MLPredictor()
            self._predictor.load()
        return self._predictor

    def filter(self, df):
        if df.empty:
            return df
        df = df.copy()
        if 'name' in df.columns:
            df = df[~df['name'].str.contains('ST|退市', na=False)]
        if 'close' in df.columns:
            df = df[df['close'] > 0]
        if 'amount' in df.columns:
            df = df[df['amount'] >= 100000]
        return df.reset_index(drop=True)

    def score(self, df):
        if df.empty:
            return df
        df = df.copy()
        try:
            predictor = self._get_predictor()
            X, _ = predictor.prepare_data(df)
            probs = predictor.predict(X)
            df['total_score'] = probs.values * 20
        except Exception:
            df['total_score'] = 0.0
        df = df[df['total_score'] >= 14]  # Higher threshold: only high-confidence predictions
        return df.sort_values('total_score', ascending=False).reset_index(drop=True)

    def generate_signals_vectorized(self, df):
        if df.empty or 'trade_date' not in df.columns:
            return pd.DataFrame()
        try:
            scored = self.score(df)
            if scored.empty or 'total_score' not in scored.columns:
                return pd.DataFrame()
            scored['signal'] = (scored['total_score'] >= 15).astype(int)
            return scored.pivot_table(
                index='trade_date', columns='ts_code', values='signal', fill_value=0
            ).astype(int)
        except Exception:
            return pd.DataFrame()
