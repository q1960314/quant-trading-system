"""
数据抓取器 v3 — 五维度全量/增量
个股+指数+板块行业+市场情绪+宏观
"""
import os, time, pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.logger import get_logger
from utils.calendar import TradeCalendar
from .manager import DataSourceManager
from .adapter import DataAdapter
from config.settings import (
    ALLOWED_MARKET, LOCAL_DATA_DIR, LOCAL_GLOBAL_DIR,
    START_DATE, END_DATE, FETCH_OPTIMIZATION, TUSHARE_TOKEN, TUSHARE_API_URL
)

logger = get_logger("fetcher")

MAJOR_INDICES = ['000001.SH','399001.SZ','399006.SZ','000688.SH','000016.SH','000300.SH','000905.SH']


class DataFetcher:
    def __init__(self):
        self.mgr = DataSourceManager(token=TUSHARE_TOKEN, api_url=TUSHARE_API_URL)
        self.cal = TradeCalendar()
        cal_df = self.mgr.get_trade_calendar()
        if cal_df is not None and not cal_df.empty:
            self.cal.load_from_source(type('',(),{'get_trade_calendar':lambda:cal_df})())
        for d in [LOCAL_DATA_DIR, LOCAL_GLOBAL_DIR]:
            os.makedirs(d, exist_ok=True)

    def fetch_full(self, start=None, end=None, markets=None):
        start, end = start or START_DATE, end or END_DATE
        markets = markets or ALLOWED_MARKET
        s, e = start.replace('-',''), end.replace('-','')
        logger.info(f"=== 全量抓取: {start} ~ {end} ===")

        # 0. 股票列表
        all_s = []
        for m in markets:
            df = self.mgr.get_stock_list(m)
            if df is not None and not df.empty: all_s.append(df)
        if not all_s: return False
        stocks = pd.concat(all_s, ignore_index=True).drop_duplicates(subset=['ts_code'])
        stocks.to_csv(os.path.join(LOCAL_GLOBAL_DIR,'stock_basic.csv'), index=False, encoding='utf-8-sig')
        codes = stocks['ts_code'].tolist()
        logger.info(f"股票列表: {len(codes)}只")

        # 1. 个股日线
        self._fetch_stocks(codes, s, e)

        # 2. 指数
        self._fetch_indices(s, e)

        # 3. 板块行业
        self._fetch_sectors(s, e)

        # 4. 市场情绪
        self._fetch_sentiment(s, e)

        # 5. 宏观
        self._fetch_macro(s, e)

        logger.info("=== 全量抓取完成 ===")
        return True

    def fetch_incremental(self):
        latest = self._detect_latest_date()
        target = self.cal.prev_trade_day(datetime.now().date())
        logger.info(f"增量: 本地{latest} → 目标{target}")
        if latest >= target:
            logger.info("数据已最新"); return True
        s = (latest + timedelta(days=1)).strftime('%Y%m%d')
        e = target.strftime('%Y%m%d')

        self._fetch_sectors(s, e)
        self._fetch_sentiment(s, e)
        self._fetch_macro(s, e)
        self._fetch_stocks(self._get_all_codes(), s, e)
        self._fetch_indices(s, e)
        return True

    # ---- 个股 ----
    def _fetch_stocks(self, codes, s, e):
        """按日期批量拉全市场日线 + daily_basic，然后按股票拆分存储"""
        ts_src = self.mgr.sources[0]
        REQUIRED_COLS = {'open', 'high', 'low', 'close', 'vol', 'amount'}

        # 遍历每一天
        start_dt = pd.Timestamp(s)
        end_dt = pd.Timestamp(e)
        days = pd.date_range(start_dt, end_dt, freq='B')  # 工作日
        daily_all, basic_all = [], []

        logger.info(f"按日批量拉取: {len(days)}个交易日 (一次拉全市场)")
        import time as _time
        for i, dt in enumerate(days):
            ds = dt.strftime('%Y%m%d')
            # Rate limiting: 120/min = 0.5s/call, 2 calls/iter = 1s/iter
            _time.sleep(0.6)
            for attempt in range(3):
                try:
                    dd = ts_src.daily(start_date=dt)
                    if dd is not None and not dd.empty:
                        daily_all.append(dd)
                    break
                except:
                    if attempt < 2:
                        _time.sleep(2)  # Wait and retry on rate limit
            try:
                db = ts_src._try('daily_basic', trade_date=ds)
                if db is not None and not db.empty:
                    basic_all.append(db)
            except: pass
            if (i+1) % 50 == 0:
                logger.info(f"  {i+1}/{len(days)} | 累计 {sum(len(x) for x in daily_all)} 条")

        if not daily_all:
            logger.error("未拉到任何日线数据！")
            return

        df_all = pd.concat(daily_all, ignore_index=True)
        logger.info(f"全量日线: {len(df_all)}行, {df_all['ts_code'].nunique()}只股票")

        # 合并 daily_basic
        if basic_all:
            df_basic = pd.concat(basic_all, ignore_index=True)
            if 'trade_date' in df_all.columns and 'trade_date' in df_basic.columns:
                df_all['trade_date'] = df_all['trade_date'].astype(str)
                df_basic['trade_date'] = df_basic['trade_date'].astype(str)
                keep = [c for c in df_basic.columns if c not in df_all.columns or c in ['ts_code','trade_date']]
                df_all = df_all.merge(df_basic[keep], on=['ts_code','trade_date'], how='left')

        # 按股票拆分存储
        df_all = DataAdapter.normalize_daily(df_all)
        ok = 0
        for code, grp in df_all.groupby('ts_code'):
            if code not in codes:
                continue  # 只保存配置的板块内的股票
            if len(grp) < 5:
                continue
            missing = REQUIRED_COLS - set(grp.columns)
            if missing:
                continue
            sp = os.path.join(LOCAL_DATA_DIR, code)
            os.makedirs(sp, exist_ok=True)
            fp = os.path.join(sp, 'daily.csv')
            if os.path.exists(fp):
                ex = pd.read_csv(fp, dtype={'trade_date': str})
                grp['trade_date'] = grp['trade_date'].astype(str)
                grp = pd.concat([ex, grp], ignore_index=True).drop_duplicates(subset=['trade_date'], keep='last')
            grp.to_csv(fp, index=False, encoding='utf-8-sig')
            ok += 1

        logger.info(f"个股完成: {ok}/{len(codes)}只有效数据")

    # ---- 指数 ----
    def _fetch_indices(self, s, e):
        logger.info("指数数据...")
        all_idx = []
        for code in MAJOR_INDICES:
            df = self.mgr._try('index_daily', code, s, e)
            if df is not None and not df.empty:
                df['ts_code'] = code
                all_idx.append(df)
        if all_idx:
            pd.concat(all_idx).to_csv(os.path.join(LOCAL_GLOBAL_DIR,'index_daily.csv'), index=False, encoding='utf-8-sig')
            logger.info(f"  指数日线: {len(all_idx)}个")

        # index_dailybasic (估值)
        try:
            from .tushare_source import TushareSource
            ts = TushareSource()
            basic = ts._call(ts._pro.index_dailybasic, trade_date=e)
            if not basic.empty:
                basic.to_csv(os.path.join(LOCAL_GLOBAL_DIR,'index_dailybasic.csv'), index=False, encoding='utf-8-sig')
                logger.info(f"  指数估值: {len(basic)}条")
        except: pass

    # ---- 板块/行业 ----
    def _fetch_sectors(self, s, e):
        logger.info("板块行业数据...")
        from .tushare_source import TushareSource
        ts = TushareSource()

        # 申万行业日线
        try:
            sw = ts.sw_daily('801010.SI', s, e)
            if not sw.empty:
                sw.to_csv(os.path.join(LOCAL_GLOBAL_DIR,'sw_daily.csv'), index=False, encoding='utf-8-sig')
                logger.info(f"  申万行业: {len(sw)}条")
        except: pass

        # 行业资金流向
        all_mf = []
        for i in range((pd.Timestamp(e)-pd.Timestamp(s)).days + 1):
            dt = (pd.Timestamp(s)+timedelta(days=i)).strftime('%Y%m%d')
            try:
                mf = ts.moneyflow_ind_dc(dt)
                if mf is not None and not mf.empty:
                    mf['trade_date'] = dt
                    all_mf.append(mf)
            except: pass
        if all_mf:
            pd.concat(all_mf).to_csv(os.path.join(LOCAL_GLOBAL_DIR,'moneyflow_ind.csv'), index=False, encoding='utf-8-sig')
            logger.info(f"  行业资金: {sum(len(x) for x in all_mf)}条")

    # ---- 市场情绪 ----
    def _fetch_sentiment(self, s, e):
        logger.info("市场情绪数据...")
        from .tushare_source import TushareSource
        ts = TushareSource()
        all_tl, all_ml, all_ls, all_hgt = [], [], [], []

        for i in range((pd.Timestamp(e)-pd.Timestamp(s)).days + 1):
            dt = (pd.Timestamp(s)+timedelta(days=i)).strftime('%Y%m%d')
            try:
                df = ts.top_list(dt)
                if df is not None and not df.empty: df['trade_date']=dt; all_tl.append(df)
            except: pass
            try:
                df = ts.top_inst(dt)
                if df is not None and not df.empty: df['trade_date']=dt; all_ml.append(df)
            except: pass
            try:
                df = ts.hsgt_top10(dt)
                if df is not None and not df.empty: all_hgt.append(df)
            except: pass

        if all_tl:
            pd.concat(all_tl).to_csv(os.path.join(LOCAL_GLOBAL_DIR,'top_list.csv'), index=False, encoding='utf-8-sig')
            logger.info(f"  龙虎榜: {sum(len(x) for x in all_tl)}条")
        if all_ml:
            pd.concat(all_ml).to_csv(os.path.join(LOCAL_GLOBAL_DIR,'top_inst.csv'), index=False, encoding='utf-8-sig')
        if all_hgt:
            pd.concat(all_hgt).to_csv(os.path.join(LOCAL_GLOBAL_DIR,'hsgt_top10.csv'), index=False, encoding='utf-8-sig')
            logger.info(f"  北向资金: {sum(len(x) for x in all_hgt)}条")

        # margin
        try:
            mg = ts.margin()
            if not mg.empty:
                mg.to_csv(os.path.join(LOCAL_GLOBAL_DIR,'margin.csv'), index=False, encoding='utf-8-sig')
                logger.info(f"  融资融券: {len(mg)}条")
        except: pass

    # ---- 宏观 ----
    def _fetch_macro(self, s, e):
        logger.info("宏观数据...")
        from .tushare_source import TushareSource
        ts = TushareSource()
        for name, method in [('shibor', ts.shibor), ('lpr', ts.shibor_lpr),
                              ('pmi', ts.cn_pmi), ('cpi', ts.cn_cpi), ('ppi', ts.cn_ppi),
                              ('money', ts.cn_m)]:
            try:
                df = method()
                if df is not None and not df.empty:
                    df.to_csv(os.path.join(LOCAL_GLOBAL_DIR, f'macro_{name}.csv'), index=False, encoding='utf-8-sig')
                    logger.info(f"  {name}: {len(df)}条")
            except: pass

    # ---- 辅助 ----
    def _detect_latest_date(self):
        latest = []
        if os.path.isdir(LOCAL_DATA_DIR):
            for d in os.listdir(LOCAL_DATA_DIR):
                fp = os.path.join(LOCAL_DATA_DIR, d, 'daily.csv')
                if os.path.exists(fp) and os.path.getsize(fp) > 1024:
                    try:
                        dd = pd.read_csv(fp, dtype={'trade_date':str})
                        dts = pd.to_datetime(dd['trade_date'], format='%Y%m%d', errors='coerce').dropna()
                        if not dts.empty: latest.append(dts.max())
                    except: pass
        return max(latest) if latest else pd.Timestamp('2019-12-31')

    def _get_all_codes(self):
        bp = os.path.join(LOCAL_GLOBAL_DIR, 'stock_basic.csv')
        if os.path.exists(bp):
            return pd.read_csv(bp, dtype={'ts_code':str})['ts_code'].tolist()
        return []
