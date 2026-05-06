"""基本面因子组: PE/PB/ROE/增速/负债率"""
import pandas as pd; import numpy as np
from .registry import FactorRegistry, FactorBase

@FactorRegistry.register
class PE_TTM(FactorBase):
    name="pe_ttm"; category="fundamental"; desc="滚动市盈率"
    def compute(self, df):
        return df['pe_ttm'] if 'pe_ttm' in df.columns else pd.Series(np.nan, index=df.index)

@FactorRegistry.register
class PB(FactorBase):
    name="pb"; category="fundamental"; desc="市净率"
    def compute(self, df):
        return df['pb'] if 'pb' in df.columns else pd.Series(np.nan, index=df.index)

@FactorRegistry.register
class ROE(FactorBase):
    name="roe"; category="fundamental"; desc="净资产收益率"
    def compute(self, df):
        return df['roe'] if 'roe' in df.columns else pd.Series(np.nan, index=df.index)

@FactorRegistry.register
class FloatMarketCap(FactorBase):
    name="float_market_cap"; category="fundamental"; desc="流通市值(亿)"
    def compute(self, df):
        if 'float_market_cap' in df.columns: return df['float_market_cap']
        if 'circ_mv' in df.columns: return df['circ_mv']/10000
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register
class DebtRatio(FactorBase):
    name="debt_ratio"; category="fundamental"; desc="资产负债率"
    def compute(self, df):
        return df['debt_to_assets'] if 'debt_to_assets' in df.columns else pd.Series(np.nan, index=df.index)

@FactorRegistry.register
class GrossMargin(FactorBase):
    name="gross_margin"; category="fundamental"; desc="毛利率"
    def compute(self, df):
        return df['grossprofit_margin'] if 'grossprofit_margin' in df.columns else pd.Series(np.nan, index=df.index)

@FactorRegistry.register
class NetMargin(FactorBase):
    name="net_margin"; category="fundamental"; desc="净利率"
    def compute(self, df):
        return df['netprofit_margin'] if 'netprofit_margin' in df.columns else pd.Series(np.nan, index=df.index)

# ---- v2 新增因子 ----

@FactorRegistry.register
class EP_TTM(FactorBase):
    name="ep_ttm"; category="fundamental"; desc="盈利收益率=1/PE_TTM"
    def compute(self, df):
        pe = df['pe_ttm'] if 'pe_ttm' in df.columns else pd.Series(np.nan, index=df.index)
        return 1.0 / pe.replace(0, np.nan)

@FactorRegistry.register
class PEG(FactorBase):
    name="peg"; category="fundamental"; desc="PEG=PE/增长率"
    def compute(self, df):
        pe = df['pe_ttm'] if 'pe_ttm' in df.columns else pd.Series(np.nan, index=df.index)
        grow = df['profit_growth'] if 'profit_growth' in df.columns else pd.Series(0.1, index=df.index)
        return pe / grow.replace(0, np.nan)

@FactorRegistry.register
class TotalMarketCap(FactorBase):
    name="total_market_cap"; category="fundamental"; desc="总市值(亿)"
    def compute(self, df):
        if 'total_mv' in df.columns: return df['total_mv'] / 10000
        if 'total_market_cap' in df.columns: return df['total_market_cap']
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register
class BpRelativity(FactorBase):
    name="bp_relativity"; category="fundamental"; desc="PB行业分位数"
    def compute(self, df):
        if 'pb' not in df.columns: return pd.Series(np.nan, index=df.index)
        return df.groupby('industry')['pb'].transform(
            lambda x: x.rank(pct=True)
        ) if 'industry' in df.columns else pd.Series(0.5, index=df.index)

@FactorRegistry.register
class DivYield(FactorBase):
    name="div_yield"; category="fundamental"; desc="股息率"
    def compute(self, df):
        return df['dv_ratio'] if 'dv_ratio' in df.columns else pd.Series(0, index=df.index)
