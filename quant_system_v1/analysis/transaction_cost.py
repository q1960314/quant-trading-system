"""
Realistic transaction cost model for A-share market.

Costs: commission + stamp tax + slippage + market impact + queue probability.
Board-specific fees: Shanghai transfer fee (0.001%), Shenzhen/ChiNext (0).
"""
import pandas as pd
import numpy as np


class TransactionCostModel:
    def __init__(self, commission_rate=0.00025, min_commission=5.0,
                 stamp_tax_rate=0.001, slippage_rate=0.001,
                 impact_base=0.001, impact_max=0.01):
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage_rate = slippage_rate
        self.impact_base = impact_base
        self.impact_max = impact_max

    def buy_cost(self, price, shares, daily_amount=None):
        value = price * shares
        commission = max(value * self.commission_rate, self.min_commission)
        slippage = value * self.slippage_rate
        impact = value * self._impact_rate(value, daily_amount)
        return commission + slippage + impact

    def sell_cost(self, price, shares, daily_amount=None):
        value = price * shares
        commission = max(value * self.commission_rate, self.min_commission)
        stamp_tax = value * self.stamp_tax_rate
        slippage = value * self.slippage_rate
        impact = value * self._impact_rate(value, daily_amount)
        return commission + stamp_tax + slippage + impact

    def _impact_rate(self, trade_value, daily_amount):
        if daily_amount is None or daily_amount <= 0:
            return self.impact_base
        ratio = trade_value / daily_amount
        result = self.impact_base * (1 + ratio * 100)
        return min(result, self.impact_max)

    def transfer_fee(self, code, shares):
        code_str = str(code).zfill(6)
        if code_str.startswith('6'):
            return shares * 0.001
        return 0.0


class LimitUpQueueModel:
    """Probability of successfully buying at limit-up price.

    Based on order-to-float ratio. Higher ratio = lower fill probability.
    """

    def fill_probability(self, order_amount, float_cap):
        if float_cap <= 0:
            return 0.5
        ratio = order_amount / float_cap
        if ratio < 0.005:
            return 0.50
        elif ratio < 0.01:
            return 0.20
        else:
            return 0.05
