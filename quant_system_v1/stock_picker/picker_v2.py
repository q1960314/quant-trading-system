"""
每日选股模块 v2 — 多策略竞争排名 + 因子评价集成

增强点:
1. 支持多策略并行选股，输出竞争排名
2. 集成因子评价反馈（仅保留高分因子）
3. 多策略共振加分（≥2个策略推荐则提升置信度）
4. 输出: 排名表 + Excel + 因子归因
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

logger = get_logger("picker_v2")


class DailyStockPickerV2:
    def __init__(self, strategy_names=None):
        self.strategy_names = strategy_names or [STRATEGY_TYPE]
        self.strategies = {n: StrategyRegistry.get(n) for n in self.strategy_names if n in StrategyRegistry.list_all()}
        self.config = STOCK_PICK_CONFIG
        self.cal = TradeCalendar()
        logger.info(f"选股器v2: {list(self.strategies.keys())}")

    def get_latest_data(self, trade_date=None):
        if trade_date is None:
            # Use the latest date actually available in data
            latest_file = os.path.join(LOCAL_GLOBAL_DIR, 'daily.csv')
            if os.path.exists(latest_file):
                try:
                    import pandas as pd
                    tmp = pd.read_csv(latest_file, nrows=1, dtype={'trade_date': str})
                    if 'trade_date' in tmp.columns and len(tmp) > 0:
                        date_str = str(tmp['trade_date'].iloc[0])
                        trade_date = pd.Timestamp(date_str)
                except Exception:
                    trade_date = self.cal.prev_trade_day()
            else:
                trade_date = self.cal.prev_trade_day()
        date_str = trade_date.strftime('%Y%m%d')
        logger.info(f"加载数据: {trade_date}")
        all_data = []
        if not os.path.isdir(LOCAL_DATA_DIR):
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
            return pd.DataFrame(), trade_date
        df = pd.concat(all_data, ignore_index=True)
        df = DataAdapter.normalize_daily(df)
        for col, val in [('break_limit_times', 0), ('up_down_times', 0), ('order_amount', 0),
                          ('inst_buy', 0), ('float_market_cap', 0), ('is_main_industry', 0)]:
            if col not in df.columns:
                df[col] = val
        logger.info(f"加载: {len(df)} stocks")
        return df, trade_date

    def pick(self, df_latest, trade_date):
        if df_latest.empty:
            return pd.DataFrame()
        from analysis.competition import merge_candidates
        picks = {}
        for sname, strategy in self.strategies.items():
            try:
                result = strategy.run(df_latest.copy())
                if not result.empty:
                    picks[sname] = result[['ts_code', 'total_score']]
            except Exception as e:
                logger.warning(f"{sname} 选股失败: {e}")
        if not picks:
            logger.warning("所有策略无选股结果")
            return pd.DataFrame()
        merged = merge_candidates(picks, resonance_bonus=0.15, top_n=self.config.get('max_output_count', 20))
        if merged.empty:
            return pd.DataFrame()
        # 附加股票名称
        if 'name' in df_latest.columns:
            name_map = df_latest.set_index('ts_code')['name'].to_dict()
            merged['name'] = merged['ts_code'].map(name_map).fillna('')
        logger.info(f"\n=== {trade_date} 选股结果 (Top {len(merged)}) ===\n" +
                     merged.to_string())
        if self.config.get('export_excel'):
            fname = f"{trade_date.strftime('%Y%m%d')}_多策略选股_v2.xlsx"
            try:
                with pd.ExcelWriter(fname, engine='openpyxl') as w:
                    merged.to_excel(w, sheet_name='选股结果', index=False)
                logger.info(f"导出: {fname}")
            except Exception as e:
                logger.warning(f"导出失败: {e}")
        return merged
