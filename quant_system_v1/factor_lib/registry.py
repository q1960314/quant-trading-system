"""
因子注册体系 v2 — 66因子，七组分类，IC/IR分析
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger("factor")


class FactorBase(ABC):
    name: str = ""; category: str = ""; desc: str = ""
    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series: pass


class FactorRegistry:
    _factors = {}

    @classmethod
    def register(cls, fc):
        inst = fc(); cls._factors[inst.name] = fc; return fc

    @classmethod
    def compute(cls, df, names=None):
        """批量计算指定因子/全部因子"""
        if names is None: names = list(cls._factors.keys())
        result = pd.DataFrame(index=df.index)
        for n in names:
            if n in cls._factors:
                try: result[n] = cls._factors[n]().compute(df)
                except Exception as e: result[n] = np.nan
        return result

    @classmethod
    def list_all(cls): return list(cls._factors.keys())

    @classmethod
    def list_by_category(cls, cat):
        return [n for n,c in cls._factors.items() if c().category == cat]

    @classmethod
    def ic_analysis(cls, df, factor_names, forward_ret_col='fwd_ret_5d'):
        """计算IC/IR: 截面Rank IC均值、IR、胜率"""
        results = []
        for n in factor_names:
            if n not in df.columns or forward_ret_col not in df.columns: continue
            ic_series = df.groupby('trade_date').apply(
                lambda g: g[n].rank(pct=True).corr(g[forward_ret_col].rank(pct=True))
            ).dropna()
            if len(ic_series) == 0: continue
            results.append({
                'factor': n, 'IC_mean': ic_series.mean(), 'IC_std': ic_series.std(),
                'IR': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
                'IC_win_rate': (ic_series > 0).mean(),
                'abs_IC_mean': ic_series.abs().mean()
            })
        return pd.DataFrame(results).sort_values('abs_IC_mean', ascending=False) if results else pd.DataFrame()


# ======================== 注册所有因子 ========================
# 延迟导入因子模块来触发注册
def _register_all():
    import factor_lib.price_factors as _
    import factor_lib.technical_factors as _
    import factor_lib.fundamental_factors as _
    import factor_lib.sentiment_factors as _
    import factor_lib.sector_macro_factors as _

_register_all()
logger.info(f"因子注册: {len(FactorRegistry._factors)}个 | {FactorRegistry.list_all()}")
