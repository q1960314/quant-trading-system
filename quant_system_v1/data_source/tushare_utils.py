"""
Tushare 数据采集工具模组
封装 tushare 接口，提供日线行情、复权因子等数据获取能力。
所有方法返回 pandas DataFrame，字段名与项目标准对齐。
"""
import time
import logging
from typing import Optional

import pandas as pd
import tushare as ts

from .base import DataSourceBase, STANDARD_COLUMNS_DAILY
from .adapter import DataAdapter

logger = logging.getLogger(__name__)


class TushareClient(DataSourceBase):
    """Tushare 数据源客户端

    封装 tushare pro 接口，统一返回标准字段 DataFrame。
    内置请求频率控制，避免触发 tushare 限频。

    Parameters
    ----------
    api_key : str
        Tushare pro API token，从 https://tushare.pro 获取。
    rate_limit : float, optional
        两次请求之间的最小间隔秒数，默认 0.3s（适应普通积分用户）。
    """

    name = "tushare"
    priority = 1  # 主数据源，优先级高

    # 每页最大记录数（tushare 上限 5000）
    PAGE_SIZE = 5000

    def __init__(self, api_key: str, rate_limit: float = 0.3) -> None:
        self._api_key = api_key
        self._rate_limit = rate_limit
        self._last_request_time: float = 0.0
        ts.set_token(api_key)
        self._pro = ts.pro_api()

    # ======================== 限频控制 ========================

    def _throttle(self) -> None:
        """简易限频：确保两次请求间隔不低于 rate_limit"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_request_time = time.time()

    # ======================== 核心数据接口 ========================

    def get_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """获取日线行情数据（前复权友好版，不含复权）

        返回字段对齐 STANDARD_COLUMNS_DAILY:
        ts_code, trade_date, open, high, low, close, vol, amount,
        pct_chg, turnover_ratio, pre_close

        Parameters
        ----------
        ts_code : str
            股票代码，如 '000001.SZ'
        start_date : str
            起始日期，格式 'YYYYMMDD'
        end_date : str
            结束日期，格式 'YYYYMMDD'

        Returns
        -------
        pd.DataFrame
            日线行情，按 trade_date 升序排列。查询失败返回空 DataFrame。
        """
        self._throttle()
        try:
            df = self._pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )
            if df is None or df.empty:
                logger.warning("get_daily 返回空数据: ts_code=%s, %s~%s", ts_code, start_date, end_date)
                return pd.DataFrame(columns=STANDARD_COLUMNS_DAILY)

            # 获取每日指标补充换手率
            self._throttle()
            daily_basic = self._pro.daily_basic(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,turnover_rate,pe,pb',
            )
            if daily_basic is not None and not daily_basic.empty:
                df = df.merge(
                    daily_basic[['ts_code', 'trade_date', 'turnover_rate']],
                    on=['ts_code', 'trade_date'],
                    how='left',
                )
                df.rename(columns={'turnover_rate': 'turnover_ratio'}, inplace=True)

            # 标准化字段
            df = DataAdapter.normalize_daily(df)
            df = df.sort_values('trade_date').reset_index(drop=True)
            return df

        except Exception as e:
            logger.error("get_daily 异常: %s", e)
            return pd.DataFrame(columns=STANDARD_COLUMNS_DAILY)

    def get_adj_factor(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """获取复权因子

        返回字段: ts_code, trade_date, adj_factor

        复权计算公式:
            前复权价 = 原价 × (最新adj_factor / 当日adj_factor)

        Parameters
        ----------
        ts_code : str
            股票代码，如 '000001.SZ'
        start_date : str
            起始日期，格式 'YYYYMMDD'
        end_date : str
            结束日期，格式 'YYYYMMDD'

        Returns
        -------
        pd.DataFrame
            复权因子表，按 trade_date 升序排列。查询失败返回空 DataFrame。
        """
        self._throttle()
        try:
            df = self._pro.adj_factor(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )
            if df is None or df.empty:
                logger.warning("get_adj_factor 返回空数据: ts_code=%s, %s~%s", ts_code, start_date, end_date)
                return pd.DataFrame(columns=['ts_code', 'trade_date', 'adj_factor'])

            # 确保类型
            df['adj_factor'] = pd.to_numeric(df['adj_factor'], errors='coerce')
            if 'trade_date' in df.columns:
                df['trade_date'] = df['trade_date'].astype(str).str.replace('-', '')

            df = df.sort_values('trade_date').reset_index(drop=True)
            return df

        except Exception as e:
            logger.error("get_adj_factor 异常: %s", e)
            return pd.DataFrame(columns=['ts_code', 'trade_date', 'adj_factor'])

    # ======================== 其他必要接口 ========================

    def get_stock_list(self, market: str = "主板") -> pd.DataFrame:
        """获取股票列表

        Parameters
        ----------
        market : str, optional
            市场筛选，默认 '主板'

        Returns
        -------
        pd.DataFrame
            字段: ts_code, symbol, name, area, industry, market, list_date
        """
        self._throttle()
        try:
            df = self._pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,area,industry,market,list_date',
            )
            if df is not None and not df.empty and market:
                df = df[df['market'] == market]
            return df.reset_index(drop=True) if df is not None and not df.empty else pd.DataFrame()
        except Exception as e:
            logger.error("get_stock_list 异常: %s", e)
            return pd.DataFrame()

    def get_limit_list(self, trade_date: str) -> pd.DataFrame:
        """获取涨跌停列表（单日）

        Parameters
        ----------
        trade_date : str
            交易日期，格式 'YYYYMMDD'

        Returns
        -------
        pd.DataFrame
            标准化后的涨跌停数据
        """
        self._throttle()
        try:
            df = self._pro.limit_list(
                trade_date=trade_date,
                limit_type='',
            )
            if df is not None and not df.empty:
                df = DataAdapter.normalize_limit_list(df)
            return df.reset_index(drop=True) if df is not None and not df.empty else pd.DataFrame()
        except Exception as e:
            logger.error("get_limit_list 异常: %s", e)
            return pd.DataFrame()

    def get_daily_basic(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """获取每日指标（换手率/量比/市盈率等）

        Parameters
        ----------
        ts_code : str
            股票代码
        start_date : str
            起始日期 'YYYYMMDD'
        end_date : str
            结束日期 'YYYYMMDD'

        Returns
        -------
        pd.DataFrame
            每日指标数据
        """
        self._throttle()
        try:
            df = self._pro.daily_basic(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )
            return df if df is not None and not df.empty else pd.DataFrame()
        except Exception as e:
            logger.error("get_daily_basic 异常: %s", e)
            return pd.DataFrame()

    def ping(self) -> bool:
        """检查 Tushare 连接是否正常

        Returns
        -------
        bool
            True 表示连接正常
        """
        try:
            self._throttle()
            df = self._pro.trade_cal(exchange='SSE', is_open='1', limit=1)
            return df is not None and not df.empty
        except Exception as e:
            logger.error("ping 失败: %s", e)
            return False


# ======================== 本地测试 ========================
if __name__ == '__main__':
    import os
    import sys

    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # 从环境变量或命令行参数获取 API Key
    api_key = os.environ.get('TUSHARE_TOKEN', '')
    if len(sys.argv) > 1:
        api_key = sys.argv[1]

    if not api_key:
        print("用法: python tushare_utils.py <TUSHARE_API_KEY>")
        print("或设置环境变量 TUSHARE_TOKEN")
        sys.exit(1)

    client = TushareClient(api_key)

    # 1) 连通性测试
    print("=" * 50)
    print("1. 连通性测试 (ping)")
    ok = client.ping()
    print(f"   结果: {'✅ 正常' if ok else '❌ 失败'}")

    if not ok:
        print("   连接失败，跳过后续测试")
        sys.exit(1)

    # 2) 日线数据测试
    print("=" * 50)
    print("2. 日线数据测试 (get_daily)")
    df_daily = client.get_daily('000001.SZ', '20260101', '20260506')
    print(f"   记录数: {len(df_daily)}")
    if not df_daily.empty:
        print(f"   字段: {list(df_daily.columns)}")
        print(f"   最新3条:\n{df_daily.tail(3).to_string()}")

    # 3) 复权因子测试
    print("=" * 50)
    print("3. 复权因子测试 (get_adj_factor)")
    df_adj = client.get_adj_factor('000001.SZ', '20260101', '20260506')
    print(f"   记录数: {len(df_adj)}")
    if not df_adj.empty:
        print(f"   字段: {list(df_adj.columns)}")
        print(f"   最新3条:\n{df_adj.tail(3).to_string()}")

    # 4) 股票列表测试
    print("=" * 50)
    print("4. 股票列表测试 (get_stock_list)")
    df_stocks = client.get_stock_list()
    print(f"   记录数: {len(df_stocks)}")
    if not df_stocks.empty:
        print(f"   字段: {list(df_stocks.columns)}")
        print(f"   前3条:\n{df_stocks.head(3).to_string()}")

    print("=" * 50)
    print("测试完成 ✅")
