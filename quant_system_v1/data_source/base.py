"""
数据源抽象基类
所有数据源必须实现这些接口。统一返回 pandas DataFrame，统一字段名。
"""
from abc import ABC, abstractmethod
import pandas as pd


# ======================== 标准字段映射 ========================
# 无论数据来自哪个源，最终输出都用这些列名
STANDARD_COLUMNS_DAILY = [
    'ts_code', 'trade_date', 'open', 'high', 'low', 'close',
    'vol', 'amount', 'pct_chg', 'turnover_ratio', 'pre_close'
]

STANDARD_COLUMNS_LIMIT = [
    'ts_code', 'trade_date', 'name', 'close', 'pct_chg',
    'limit_status',       # 'U'=涨停, 'D'=跌停, ''=正常
    'order_amount',        # 封单金额(万元)
    'float_market_cap',   # 流通市值(亿)
    'up_down_times',       # 连板高度
    'break_limit_times',   # 炸板次数
    'first_limit_time',    # 首次封板时间
    'turnover_ratio',
]

STANDARD_COLUMNS_FINANCIAL = [
    'ts_code', 'end_date', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
    'roe', 'roa', 'grossprofit_margin', 'netprofit_margin',
    'revenue_yoy', 'profit_yoy', 'debt_to_assets',
]


class DataSourceBase(ABC):
    """数据源抽象基类"""

    name: str = "base"
    priority: int = 0  # 越小越优先

    @abstractmethod
    def get_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """日线行情 (OHLCV + 换手率 + 涨跌幅)"""
        pass

    @abstractmethod
    def get_stock_list(self, market: str = "主板") -> pd.DataFrame:
        """获取股票列表 [ts_code, symbol, name, area, industry, market, list_date]"""
        pass

    @abstractmethod
    def get_limit_list(self, trade_date: str) -> pd.DataFrame:
        """涨跌停列表（单日）"""
        pass

    def get_daily_basic(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """每日指标（换手率/量比/市盈率等），可选实现"""
        return pd.DataFrame()

    def get_financial(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """财务数据，可选实现"""
        return pd.DataFrame()

    def get_moneyflow(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """资金流向，可选实现"""
        return pd.DataFrame()

    def get_top_list(self, trade_date: str) -> pd.DataFrame:
        """龙虎榜，可选实现"""
        return pd.DataFrame()

    def get_trade_calendar(self) -> pd.DataFrame:
        """交易日历，可选实现"""
        return pd.DataFrame()

    def get_spot(self) -> pd.DataFrame:
        """实时行情（盘中用），可选实现"""
        return pd.DataFrame()

    def ping(self) -> bool:
        """检查数据源是否可用"""
        return True
