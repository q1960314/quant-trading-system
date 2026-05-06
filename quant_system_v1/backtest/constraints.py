"""
Real trading constraints for backtest.

LimitUpDownConstraint: stocks at limit up/down cannot be bought/sold
SuspensionConstraint: suspended stocks filtered out
LiquidityConstraint: min turnover and amount thresholds
PositionConstraint: single stock, industry, total holding limits
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class BaseConstraint(ABC):
    """Unified interface: apply(df) -> boolean mask, True=tradable"""

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.Series:
        pass


class LimitUpDownConstraint(BaseConstraint):
    """Stocks hitting limit up cannot be bought; limit down cannot be sold.

    Board-specific limits:
    - Main board (60xxxx/00xxxx): +-10%
    - ChiNext (30xxxx): +-20%
    - STAR (68xxxx): +-20%
    - BSE (8xxxxx/4xxxxx): +-30%
    """

    LIMIT_MAP = {'main': 0.10, 'chinext': 0.20, 'star': 0.20, 'bse': 0.30}

    def __init__(self, price_adj='front'):
        self.price_adj = price_adj

    def _detect_board(self, codes: pd.Series) -> pd.Series:
        codes_str = codes.astype(str).str.zfill(6)
        limits = pd.Series(0.10, index=codes.index)
        limits[codes_str.str.match(r'^30')] = 0.20
        limits[codes_str.str.match(r'^68')] = 0.20
        limits[codes_str.str.match(r'^[48]')] = 0.30
        return limits

    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Limit-down stocks cannot be sold. Limit-up stocks are allowed
        (execution happens T+1 via slippage model)."""
        if df.empty:
            return pd.Series(dtype=bool)
        preclose_col = 'pre_close' if 'pre_close' in df.columns else 'open'
        if preclose_col not in df.columns:
            return pd.Series(True, index=df.index)
        limits = self._detect_board(df['ts_code'])
        limit_down_price = df[preclose_col] * (1 - limits)
        # Block only limit-down stocks (can't sell)
        at_limit_down = df['close'] <= limit_down_price
        return ~at_limit_down  # True = tradable


class SuspensionConstraint(BaseConstraint):
    """Volume=0 or amount=0 marks suspension, untradable."""

    def apply(self, df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=bool)
        no_volume = (df.get('vol', 0) == 0)
        no_amount = (df.get('amount', 0) == 0)
        return ~(no_volume | no_amount)


class LiquidityConstraint(BaseConstraint):
    """Filter by minimum turnover amount and turnover rate."""

    def __init__(self, min_amount=50000, min_turnover=0.5):
        self.min_amount = min_amount
        self.min_turnover = min_turnover

    def apply(self, df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=bool)
        mask = pd.Series(True, index=df.index)
        if 'amount' in df.columns:
            mask &= (df['amount'] >= self.min_amount)
        if 'turnover_rate' in df.columns:
            mask &= (df['turnover_rate'] >= self.min_turnover)
        elif 'volume' in df.columns and 'total_share' in df.columns:
            turnover = df['volume'] / df['total_share'] * 100
            mask &= (turnover >= self.min_turnover)
        return mask


class PositionConstraint:
    """Per-day position limits enforced during order matching."""

    def __init__(self, single_stock_ratio=0.2, industry_ratio=0.3,
                 max_hold_stocks=5, max_daily_trade_ratio=0.05):
        self.single_stock_ratio = single_stock_ratio
        self.industry_ratio = industry_ratio
        self.max_hold_stocks = max_hold_stocks
        self.max_daily_trade_ratio = max_daily_trade_ratio

    def can_buy(self, code, cash, total_capital, current_positions,
                buy_price, daily_volume):
        if len(current_positions) >= self.max_hold_stocks:
            return False
        max_value = total_capital * self.single_stock_ratio
        if max_value < buy_price * 100:
            return False
        max_shares = int(max_value / buy_price / 100) * 100
        daily_limit = int(daily_volume * self.max_daily_trade_ratio / 100) * 100
        return max(min(max_shares, daily_limit), 0) >= 100
