"""
策略基类 — 所有策略继承此类，实现 filter() 和 score()
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class Signal:
    ts_code: str; signal_type: str; price: float  # buy / sell
    shares: int = 0; confidence: float = 0; reason: str = ""
    score: float = 0; metadata: Dict = field(default_factory=dict)

@dataclass
class Position:
    ts_code: str; shares: int; cost_price: float
    current_price: float = 0; buy_date: str = ""; buy_idx: int = 0
    total_score: float = 0; industry: str = ""

    @property
    def pnl_pct(self):
        if self.cost_price <= 0: return 0
        return (self.current_price - self.cost_price) / self.cost_price

@dataclass
class RiskParams:
    stop_loss: float = 0.06; stop_profit: float = 0.12
    max_hold_days: int = 3; max_positions: int = 5
    position_ratio: float = 0.2; slippage_rate: float = 0.005


class StrategyBase(ABC):
    name: str = "base"; version: str = "1.0"

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.risk = RiskParams(**self.config.get('risk', {})) if 'risk' in self.config else RiskParams()

    @abstractmethod
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """返回带 total_score 列的 DataFrame"""
        pass

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """完整管线：过滤→评分→排序"""
        df = self.filter(df)
        if df.empty: return df
        df = self.score(df)
        if df.empty: return df
        return df.sort_values('total_score', ascending=False).reset_index(drop=True)

    def generate_signals_vectorized(self, df: pd.DataFrame) -> 'pd.DataFrame':
        """可选：返回 (T, N) 信号矩阵。1=buy, -1=sell, 0=hold。
        默认回退到逐行 filter()+score()。"""
        import pandas as pd
        return pd.DataFrame()


class StrategyRegistry:
    _strategies: Dict[str, type] = {}

    @classmethod
    def register(cls, strategy_cls):
        inst = strategy_cls()
        cls._strategies[inst.name] = strategy_cls
        return strategy_cls

    @classmethod
    def get(cls, name: str, config: Dict = None):
        if name not in cls._strategies:
            raise ValueError(f"策略 '{name}' 未注册。可用: {list(cls._strategies.keys())}")
        return cls._strategies[name](config)

    @classmethod
    def list_all(cls):
        return list(cls._strategies.keys())
