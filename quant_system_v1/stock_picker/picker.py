"""
每日选股模块 — 从最新交易日数据选股
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from utils.logger import get_logger
from utils.calendar import TradeCalendar
from strategy import StrategyRegistry
from data_source import DataAdapter
from config.settings import STOCK_PICK_CONFIG, LOCAL_DATA_DIR, LOCAL_GLOBAL_DIR, STRATEGY_TYPE

logger = get_logger("picker")


class DailyStockPicker:
    def __init__(self, strategy_name=None):
        self.strategy_name = strategy_name or STRATEGY_TYPE
        self.strategy = StrategyRegistry.get(self.strategy_name)
        self.config = STOCK_PICK_CONFIG
        self.cal = TradeCalendar()
        logger.info(f"选股器初始化: {self.strategy_name}")

    def get_latest_data(self, trade_date=None):
        """加载最新交易日数据"""
        if trade_date is None:
            trade_date = self.cal.prev_trade_day()

        date_str = trade_date.strftime('%Y%m%d')
        logger.info(f"最新交易日: {trade_date} ({date_str})")

        all_data = []
        if not os.path.isdir(LOCAL_DATA_DIR):
            logger.error(f"数据目录不存在: {LOCAL_DATA_DIR}")
            return pd.DataFrame(), trade_date

        stock_basic = pd.DataFrame()
        bp = os.path.join(LOCAL_GLOBAL_DIR, 'stock_basic.csv')
        if os.path.exists(bp):
            stock_basic = pd.read_csv(bp, dtype={'ts_code': str})

        for d in os.listdir(LOCAL_DATA_DIR):
            dp = os.path.join(LOCAL_DATA_DIR, d)
            if os.path.isdir(dp):
                fp = os.path.join(dp, 'daily.csv')
                if os.path.exists(fp) and os.path.getsize(fp) > 1024:
                    try:
                        dd = pd.read_csv(fp, dtype={'trade_date': str})
                        dd['trade_date'] = pd.to_datetime(dd['trade_date'], format='%Y%m%d', errors='coerce')
                        dd = dd[dd['trade_date'] == trade_date]
                        if not dd.empty:
                            if not stock_basic.empty:
                                bi = stock_basic[stock_basic['ts_code'] == d]
                                if not bi.empty:
                                    dd['name'] = bi.iloc[0].get('name', '')
                                    dd['industry'] = bi.iloc[0].get('industry', '')
                            dd['ts_code'] = d
                            all_data.append(dd)
                    except Exception:
                        pass

        if not all_data:
            logger.error("未找到最新交易日数据！")
            return pd.DataFrame(), trade_date

        df = pd.concat(all_data, ignore_index=True)
        df = DataAdapter.normalize_daily(df)
        # 填充默认字段
        for col, val in [('break_limit_times', 0), ('up_down_times', 0), ('order_amount', 0),
                          ('inst_buy', 0), ('youzi_buy', 0), ('concept_count', 0),
                          ('no_reduction', 1), ('no_inquiry', 1), ('is_main_industry', 0), ('float_market_cap', 0)]:
            if col not in df.columns:
                df[col] = val

        logger.info(f"加载完成: {len(df)}只标的")
        return df, trade_date

    def pick(self, df_latest, trade_date):
        """执行选股"""
        if df_latest.empty:
            return pd.DataFrame()

        logger.info(f"执行选股: {self.strategy_name} | {trade_date}")
        result = self.strategy.run(df_latest)

        if result.empty:
            logger.warning("无符合条件的标的")
            return pd.DataFrame()

        result = result.head(self.config['max_output_count'])

        # 格式化输出
        cols_show = ['ts_code', 'name', 'total_score']
        if 'industry' in result.columns: cols_show.insert(2, 'industry')
        if 'close' in result.columns: cols_show.append('close')
        if 'turnover_ratio' in result.columns: cols_show.append('turnover_ratio')

        display = result[[c for c in cols_show if c in result.columns]].copy()
        display.columns = ['股票代码', '股票名称', '所属行业', '总评分', '收盘价', '换手率'][:len(display.columns)]
        # re-align after potential column insertion
        rename_map = {'ts_code': '股票代码', 'name': '股票名称', 'industry': '所属行业',
                       'total_score': '总评分', 'close': '收盘价', 'turnover_ratio': '换手率'}
        display = result[[c for c in cols_show if c in result.columns]].copy()
        display.columns = [rename_map.get(c, c) for c in display.columns]

        logger.info("\n" + display.to_string(index=False))

        # 导出Excel
        if self.config.get('export_excel'):
            fname = f"{trade_date.strftime('%Y%m%d')}_{self.strategy_name}_选股结果.xlsx"
            try:
                with pd.ExcelWriter(fname, engine='openpyxl') as w:
                    result.to_excel(w, sheet_name='选股结果', index=False)
                logger.info(f"选股结果已导出: {fname}")
            except Exception as e:
                logger.warning(f"Excel导出失败: {e}")

        return result
