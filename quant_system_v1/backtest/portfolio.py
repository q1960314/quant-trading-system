"""Multi-strategy portfolio allocator."""
import numpy as np


class EqualWeightAllocator:
    def allocate(self, strategies: list, capital: float) -> dict:
        w = 1.0 / max(len(strategies), 1)
        return {s: capital * w for s in strategies}


class KellyAllocator:
    def allocate(self, strategies: list, capital: float,
                 win_rates: dict, profit_loss_ratios: dict) -> dict:
        result = {}
        total_kelly = 0
        kellys = {}
        for s in strategies:
            wr = win_rates.get(s, 0.5)
            pl = profit_loss_ratios.get(s, 1.0)
            k = (wr * (pl + 1) - 1) / pl if pl > 0 else 0
            k = max(0, min(k, 0.25))
            kellys[s] = k
            total_kelly += k
        for s in strategies:
            result[s] = capital * kellys.get(s, 0) / total_kelly if total_kelly > 0 else capital / max(len(strategies), 1)
        return result


class RiskParityAllocator:
    def allocate(self, strategies: list, capital: float,
                 risk_contributions: dict, target_vol: float = 0.10) -> dict:
        result = {}
        total_inv_vol = sum(1.0 / max(v, 0.01) for v in risk_contributions.values())
        for s in strategies:
            inv_vol = 1.0 / max(risk_contributions.get(s, 0.10), 0.01)
            result[s] = capital * inv_vol / total_inv_vol if total_inv_vol > 0 else capital / max(len(strategies), 1)
        return result
