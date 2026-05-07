"""
步骤1：全量数据抓取
按接口特性分类：一次性 / 按日期 / 按股票
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
from datetime import datetime

from config.settings import LOCAL_DATA_DIR, LOCAL_GLOBAL_DIR
from data_source import DataAdapter

TOKEN = "123396cf48bacd87370d4541fe4c2c51bcacb43f32f66890650dd5bb907a"

def init_pro():
    import tushare as ts
    pro = ts.pro_api(TOKEN)
    pro._DataApi__token = TOKEN
    pro._DataApi__http_url = "http://jiaoch.site"
    return pro

def safe_call(fn, name):
    try:
        df = fn()
        if df is not None and not df.empty:
            return df
    except Exception as e:
        print(f"  [{name}] skip: {str(e)[:80]}")
    return pd.DataFrame()

def save(df, filename, subdir=''):
    p = os.path.join(LOCAL_GLOBAL_DIR, subdir, filename) if subdir else os.path.join(LOCAL_GLOBAL_DIR, filename)
    df.to_csv(p, index=False, encoding='utf-8-sig')

def main(start_date="2020-01-01", end_date="2025-05-06"):
    pro = init_pro()
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    os.makedirs(LOCAL_GLOBAL_DIR, exist_ok=True)
    days = pd.date_range(start_date, end_date, freq='B')
    total_days = len(days)
    print(f"抓取范围: {start_date} ~ {end_date} ({total_days}个交易日)")

    # ====== A. 一次性抓取 ======
    print("\n=== A. 一次性数据 ===")

    df = safe_call(lambda: pro.stock_basic(exchange='', list_status='L',
        fields='ts_code,symbol,name,area,industry,list_date,market'), 'stock_basic')
    if not df.empty: save(df, 'stock_basic.csv'); print(f"  stock_basic: {len(df)}只")

    df = safe_call(pro.ths_index, 'ths_index')
    if not df.empty: save(df, 'ths_index.csv'); print(f"  ths_index: {len(df)}个板块")

    df = safe_call(pro.tdx_index, 'tdx_index')
    if not df.empty: save(df, 'tdx_index.csv'); print(f"  tdx_index: {len(df)}个板块")

    df = safe_call(lambda: pro.tdx_member(), 'tdx_member')
    if not df.empty: save(df, 'tdx_member.csv'); print(f"  tdx_member: {len(df)}条成分")

    df = safe_call(lambda: pro.kpl_concept_cons(trade_date=end_date.replace('-','')), 'kpl_concept_cons')
    if not df.empty: save(df, 'kpl_concept_cons.csv'); print(f"  kpl_concept_cons: {len(df)}条")

    # ====== B. 按日期循环抓取 ======
    print(f"\n=== B. 按日期循环 ({total_days}天) ===")
    daily_all, basic_all, top_all, top_inst_all = [], [], [], []
    limit_d_all, limit_ths_all, limit_step_all = [], [], []
    hm_all, hot_all, mf_cnt_all, mf_ind_all, mf_hsgt_all = [], [], [], [], []

    t0 = time.time()
    for i, dt in enumerate(days):
        ds = dt.strftime('%Y%m%d')

        # P0: 核心行情
        dd = safe_call(lambda: pro.daily(trade_date=ds), f'daily_{ds}')
        if not dd.empty: daily_all.append(dd)

        db = safe_call(lambda: pro.daily_basic(trade_date=ds), f'daily_basic_{ds}')
        if not db.empty: basic_all.append(db)

        # P1: 龙虎榜
        tl = safe_call(lambda: pro.top_list(trade_date=ds), f'top_list_{ds}')
        if not tl.empty: tl['trade_date'] = ds; top_all.append(tl)

        ti = safe_call(lambda: pro.top_inst(trade_date=ds), f'top_inst_{ds}')
        if not ti.empty: ti['trade_date'] = ds; top_inst_all.append(ti)

        # P1: 涨跌停+连板
        ld = safe_call(lambda: pro.limit_list_d(trade_date=ds), f'limit_list_d_{ds}')
        if not ld.empty: ld['trade_date'] = ds; limit_d_all.append(ld)

        ls = safe_call(lambda: pro.limit_step(trade_date=ds), f'limit_step_{ds}')
        if not ls.empty: ls['trade_date'] = ds; limit_step_all.append(ls)

        # P1: 游资+热榜
        hm = safe_call(lambda: pro.hm_detail(trade_date=ds), f'hm_detail_{ds}')
        if not hm.empty: hm['trade_date'] = ds; hm_all.append(hm)

        hot = safe_call(lambda: pro.ths_hot(trade_date=ds), f'ths_hot_{ds}')
        if not hot.empty: hot['trade_date'] = ds; hot_all.append(hot)

        # P2: 资金流向
        mc = safe_call(lambda: pro.moneyflow_cnt_ths(trade_date=ds), f'mf_cnt_{ds}')
        if not mc.empty: mc['trade_date'] = ds; mf_cnt_all.append(mc)

        mi = safe_call(lambda: pro.moneyflow_ind_ths(trade_date=ds), f'mf_ind_{ds}')
        if not mi.empty: mi['trade_date'] = ds; mf_ind_all.append(mi)

        mh = safe_call(lambda: pro.moneyflow_hsgt(trade_date=ds), f'mf_hsgt_{ds}')
        if not mh.empty: mh['trade_date'] = ds; mf_hsgt_all.append(mh)

        if (i+1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (total_days - i - 1)
            print(f"  [{i+1}/{total_days}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining | 日线累计{sum(len(x) for x in daily_all)}条")

    elapsed = time.time() - t0
    print(f"  日期循环完成: {elapsed:.0f}s")

    # ====== C. 合并 + 存储 ======
    print("\n=== C. 合并存储 ===")

    # 日线：按股票拆分
    if daily_all:
        df_daily = pd.concat(daily_all, ignore_index=True)
        if basic_all:
            df_basic = pd.concat(basic_all, ignore_index=True)
            df_daily['trade_date'] = df_daily['trade_date'].astype(str)
            df_basic['trade_date'] = df_basic['trade_date'].astype(str)
            keep = [c for c in df_basic.columns if c not in df_daily.columns or c in ['ts_code','trade_date']]
            df_daily = df_daily.merge(df_basic[keep], on=['ts_code','trade_date'], how='left')

        df_daily = DataAdapter.normalize_daily(df_daily)
        stock_count = 0
        for code, grp in df_daily.groupby('ts_code'):
            grp = grp.sort_values('trade_date')
            sp = os.path.join(LOCAL_DATA_DIR, code)
            os.makedirs(sp, exist_ok=True)
            grp.to_csv(os.path.join(sp, 'daily.csv'), index=False, encoding='utf-8-sig')
            stock_count += 1
        print(f"  日线: {len(df_daily)}行 → {stock_count}只股票")

    # 全局CSV
    global_files = [
        (top_all, 'top_list.csv'),
        (top_inst_all, 'top_inst.csv'),
        (limit_d_all, 'limit_list_d.csv'),
        (limit_step_all, 'limit_step.csv'),
        (hm_all, 'hm_detail.csv'),
        (hot_all, 'ths_hot.csv'),
        (mf_cnt_all, 'moneyflow_cnt_ths.csv'),
        (mf_ind_all, 'moneyflow_ind_ths.csv'),
        (mf_hsgt_all, 'moneyflow_hsgt.csv'),
    ]
    for data_list, fname in global_files:
        if data_list:
            df = pd.concat(data_list, ignore_index=True)
            save(df, fname)
            print(f"  {fname}: {len(df)}行")

    # ====== D. 验证 ======
    print("\n=== D. 数据验证 ===")
    stock_count = total_rows = 0
    min_r, max_r = 999999, 0
    for d in os.listdir(LOCAL_DATA_DIR):
        fp = os.path.join(LOCAL_DATA_DIR, d, 'daily.csv')
        if os.path.exists(fp):
            rows = len(pd.read_csv(fp))
            stock_count += 1; total_rows += rows
            min_r = min(min_r, rows); max_r = max(max_r, rows)

    print(f"  个股: {stock_count}只, 总{total_rows}行, 单只{min_r}-{max_r}行")
    print(f"  全局文件: {len(os.listdir(LOCAL_GLOBAL_DIR))}个")
    for f in sorted(os.listdir(LOCAL_GLOBAL_DIR)):
        sz = os.path.getsize(os.path.join(LOCAL_GLOBAL_DIR, f)) / 1024
        print(f"    {f}: {sz:.0f}KB")

    print("\n[完成]")

if __name__ == '__main__':
    s = sys.argv[1] if len(sys.argv) > 1 else '2020-01-01'
    e = sys.argv[2] if len(sys.argv) > 2 else '2025-05-06'
    main(s, e)
