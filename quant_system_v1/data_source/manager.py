"""
数据源管理器 — 多源自动降级
按优先级尝试：Tushare → AKShare → 东方财富 → Baostock
"""
import pandas as pd
from .base import DataSourceBase
from .tushare_source import TushareSource
from .akshare_source import AKShareSource
from .eastmoney_source import EastMoneySource
from .baostock_source import BaostockSource


class DataSourceManager:
    def __init__(self, token="", api_url="http://jiaoch.site"):
        self.sources = []
        # 按优先级注册
        ts = TushareSource(token=token, api_url=api_url)
        if ts.ping():
            self.sources.append(ts)
            print(f"[DataSource] Tushare [OK]")
        else:
            print(f"[DataSource] Tushare [FAIL] (token未生效或网络不通)")

        for src_cls in [AKShareSource, EastMoneySource]:
            src = src_cls()
            if src.ping():
                self.sources.append(src)
                print(f"[DataSource] {src.name} [OK]")

        bao = BaostockSource()
        self.sources.append(bao)  # Baostock 总是可用的
        print(f"[DataSource] Baostock [OK] (备用)")

        print(f"[DataSource] 共 {len(self.sources)} 个可用源: {[s.name for s in self.sources]}")

        self._primary = self.sources[0] if self.sources else None

    def _try(self, method, *args, **kwargs):
        """遍历数据源，找到第一个返回非空的就停"""
        for src in self.sources:
            try:
                result = getattr(src, method)(*args, **kwargs)
                if result is not None and (not isinstance(result, pd.DataFrame) or not result.empty):
                    return result
            except Exception as e:
                print(f"[DataSource] {src.name}.{method} 失败: {e}")
                continue
        return pd.DataFrame()

    def get_daily(self, ts_code, start_date, end_date):
        return self._try('get_daily', ts_code, start_date, end_date)

    def get_stock_list(self, market="主板"):
        for src in self.sources:
            df = src.get_stock_list(market)
            if df is not None and not df.empty:
                return df
        return pd.DataFrame()

    def get_limit_list(self, trade_date):
        return self._try('get_limit_list', trade_date)

    def get_daily_basic(self, ts_code, start_date, end_date):
        return self._try('get_daily_basic', ts_code, start_date, end_date)

    def get_financial(self, ts_code, start_date, end_date):
        return self._try('get_financial', ts_code, start_date, end_date)

    def get_moneyflow(self, ts_code, start_date, end_date):
        return self._try('get_moneyflow', ts_code, start_date, end_date)

    def get_top_list(self, trade_date):
        return self._try('get_top_list', trade_date)

    def get_trade_calendar(self):
        return self._try('get_trade_calendar')

    def get_spot(self):
        """实时行情，优先东方财富（最快）"""
        for src in self.sources:
            if src.name in ('eastmoney', 'akshare'):
                try:
                    df = src.get_spot()
                    if df is not None and not df.empty:
                        return df
                except Exception:
                    continue
        return pd.DataFrame()

    @property
    def primary(self):
        return self._primary
