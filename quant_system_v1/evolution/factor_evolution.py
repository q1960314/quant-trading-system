"""
Factor evolution: regime rotation, adaptive weight optimization, lifecycle monitoring.
"""
import os, sys, json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from factor_lib.registry import FactorRegistry

logger = get_logger("factor_evo")


class FactorRotator:
    def __init__(self):
        self.regime_factors = {
            'bull': {'price': 1.5, 'sentiment': 1.3, 'volume': 1.2, 'technical': 1.0,
                     'fundamental': 0.8, 'sector': 1.0, 'macro': 0.7},
            'bear': {'fundamental': 1.5, 'macro': 1.3, 'sector': 1.0, 'technical': 1.0,
                     'price': 0.7, 'volume': 0.8, 'sentiment': 0.7},
            'normal': {'price': 1.0, 'volume': 1.0, 'technical': 1.0, 'fundamental': 1.0,
                       'sentiment': 1.0, 'sector': 1.0, 'macro': 1.0},
        }

    def get_weights(self, regime, factor_names):
        weights = self.regime_factors.get(regime, self.regime_factors['normal'])
        result = {}
        for name in factor_names:
            inst = FactorRegistry._factors.get(name)
            cat = inst().category if inst else 'price'
            result[name] = weights.get(cat, 1.0)
        return result


class AdaptiveWeightOptimizer:
    def __init__(self, history_file=None):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.history_file = history_file or os.path.join(base, 'evolution_output', 'ic_history.json')

    def update(self, factor_name, ic_value, date):
        history = self._load()
        if factor_name not in history:
            history[factor_name] = []
        history[factor_name].append({'date': date, 'ic': float(ic_value)})
        history[factor_name] = history[factor_name][-252:]
        self._save(history)

    def get_weight(self, factor_name, lookback_months=6):
        history = self._load()
        entries = history.get(factor_name, [])
        if not entries:
            return 1.0
        cutoff = (datetime.now() - timedelta(days=lookback_months * 30)).isoformat()[:10]
        recent = [e for e in entries if e['date'] >= cutoff]
        if not recent:
            return 1.0
        ics = [abs(e['ic']) for e in recent]
        avg_ic = np.mean(ics)
        if avg_ic < 0.1:
            return 0.0
        if avg_ic < 0.15:
            return 0.3
        return min(avg_ic / 0.05, 2.0)

    def _load(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {}

    def _save(self, history):
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)


class FactorLifecycleMonitor:
    def __init__(self, history_file=None):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.history_file = history_file or os.path.join(base, 'evolution_output', 'ic_history.json')

    def check(self, factor_name):
        history = self._load()
        entries = history.get(factor_name, [])
        if len(entries) < 20:
            return {'status': 'new', 'trend': 'unknown', 'action': 'KEEP'}
        ics = [e['ic'] for e in entries[-60:]]
        avg_recent = np.mean(ics[-20:])
        avg_prior = np.mean(ics[-40:-20]) if len(ics) >= 40 else avg_recent
        change_pct = (avg_recent - avg_prior) / max(abs(avg_prior), 0.001)
        if change_pct < -0.3:
            return {'status': 'CRASH', 'trend': 'down', 'change': change_pct, 'action': 'REMOVE', 'avg_ic': avg_recent}
        if change_pct < -0.15:
            return {'status': 'DECLINING', 'trend': 'down', 'change': change_pct, 'action': 'WARN', 'avg_ic': avg_recent}
        if change_pct > 0.15:
            return {'status': 'IMPROVING', 'trend': 'up', 'change': change_pct, 'action': 'BOOST', 'avg_ic': avg_recent}
        return {'status': 'STABLE', 'trend': 'flat', 'change': change_pct, 'action': 'KEEP', 'avg_ic': avg_recent}

    def _load(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {}
