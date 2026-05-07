"""
全量数据抓取 v2 — 完整35接口
用法: python scripts/fetch_all.py 2024-10-01 2024-12-31
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from datetime import datetime
from config.settings import LOCAL_DATA_DIR, LOCAL_GLOBAL_DIR
from data_source import DataAdapter

TOKEN = "123396cf48bacd87370d4541fe4c2c51bcacb43f32f66890650dd5bb907a"
SAVE_INTERVAL = 15  # 每N天保存一次进度

def init():
    import tushare as ts
    pro = ts.pro_api(TOKEN); pro._DataApi__token = TOKEN; pro._DataApi__http_url = "http://jiaoch.site"
    return pro

def call(fn, name):
    try:
        df = fn()
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception as e:
        print(f"  [{name}] {str(e)[:60]}"); return pd.DataFrame()

def save(df, name):
    df.to_csv(os.path.join(LOCAL_GLOBAL_DIR, name), index=False, encoding='utf-8-sig')

# ====================== A. 一次性数据 ======================
def fetch_once(pro):
    print("=== A. 一次性数据 ===\n")
    results = {}

    r = call(lambda: pro.stock_basic(exchange='', list_status='L',
        fields='ts_code,symbol,name,area,industry,list_date,market'),'stock_basic')
    if not r.empty: save(r, 'stock_basic.csv'); print(f"  stock_basic: {len(r)}只"); results['stock_basic'] = r

    for api, fname in [('trade_cal','trade_cal.csv'),('ths_index','ths_index.csv'),
        ('tdx_index','tdx_index.csv'),('tdx_member','tdx_member.csv')]:
        r = call(lambda a=api: getattr(pro, a)(), api)
        if not r.empty: save(r, fname); print(f"  {api}: {len(r)}行")

    r = call(lambda: pro.kpl_concept_cons(trade_date='20241231'), 'kpl_concept_cons')
    if not r.empty: save(r, 'kpl_concept_cons.csv'); print(f"  kpl_concept_cons: {len(r)}行")

    for macro_name, fn in [('shibor', lambda: pro.shibor()), ('cn_pmi', lambda: pro.cn_pmi()),
        ('cn_cpi', lambda: pro.cn_cpi()), ('cn_m', lambda: pro.cn_m())]:
        r = call(fn, macro_name)
        if not r.empty: save(r, f'macro_{macro_name}.csv'); print(f"  {macro_name}: {len(r)}行")

    return results

# ====================== B. 按日期循环 ======================
def fetch_daily(pro, start_date, end_date):
    days = pd.date_range(start_date, end_date, freq='B')
    total = len(days)
    print(f"\n=== B. 按日期循环: {total}天 ===\n")

    # 定义每个接口: (名称, 调用函数, 收集列表)
    collectors = [
        ('daily', lambda ds: pro.daily(trade_date=ds), []),
        ('daily_basic', lambda ds: pro.daily_basic(trade_date=ds), []),
        ('top_list', lambda ds: call(lambda: pro.top_list(trade_date=ds),''), []),
        ('top_inst', lambda ds: call(lambda: pro.top_inst(trade_date=ds),''), []),
        ('limit_list_d', lambda ds: call(lambda: pro.limit_list_d(trade_date=ds),''), []),
        ('limit_step', lambda ds: call(lambda: pro.limit_step(trade_date=ds),''), []),
        ('hm_detail', lambda ds: call(lambda: pro.hm_detail(trade_date=ds),''), []),
        ('ths_hot', lambda ds: call(lambda: pro.ths_hot(trade_date=ds),''), []),
        ('moneyflow_cnt_ths', lambda ds: call(lambda: pro.moneyflow_cnt_ths(trade_date=ds),''), []),
        ('moneyflow_ind_ths', lambda ds: call(lambda: pro.moneyflow_ind_ths(trade_date=ds),''), []),
        ('moneyflow_hsgt', lambda ds: call(lambda: pro.moneyflow_hsgt(trade_date=ds),''), []),
        ('stk_limit', lambda ds: pro.stk_limit(trade_date=ds), []),
        ('margin_detail', lambda ds: call(lambda: pro.margin_detail(trade_date=ds),''), []),
        ('hsgt_top10', lambda ds: call(lambda: pro.hsgt_top10(trade_date=ds),''), []),
        ('suspend_d', lambda ds: call(lambda: pro.suspend_d(trade_date=ds),''), []),
    ]

    t0 = time.time()

    for i, dt in enumerate(days):
        ds = dt.strftime('%Y%m%d')

        for _, getter, buf in collectors:
            try:
                df = getter(ds)
                if df is not None and not df.empty:
                    if 'trade_date' not in df.columns:
                        df['trade_date'] = ds
                    buf.append(df)
            except: pass
            time.sleep(0.6)  # 控制频率，避免被限流

        if (i+1) % SAVE_INTERVAL == 0 or (i+1) == total:
            elapsed = time.time() - t0
            d_count = sum(len(b) for _, _, b in collectors if b)
            day_count = len(collectors[0][2])
            print(f"  [{i+1}/{total}] {elapsed:.0f}s | 日线{sum(len(x) for x in collectors[0][2])}条 | {day_count}天有数据")

            # 保存进度
            save_intermediate(collectors)

    return collectors

def save_intermediate(collectors):
    """保存中间结果到全局CSV"""
    for name, _, buf in collectors:
        if buf:
            df = pd.concat(buf, ignore_index=True)
            save(df, f'{name}.csv')

# ====================== C. 拆分日线到个股 ======================
def split_daily():
    print("\n=== C. 拆分日线到个股 ===")
    fp = os.path.join(LOCAL_GLOBAL_DIR, 'daily.csv')
    if not os.path.exists(fp):
        print("  daily.csv不存在!"); return

    df = pd.read_csv(fp, dtype={'trade_date': str, 'ts_code': str})
    print(f"  日线: {len(df)}行")

    bp = os.path.join(LOCAL_GLOBAL_DIR, 'daily_basic.csv')
    if os.path.exists(bp):
        db = pd.read_csv(bp, dtype={'trade_date': str})
        df['trade_date'] = df['trade_date'].astype(str)
        keep = [c for c in db.columns if c not in df.columns or c in ['ts_code','trade_date']]
        df = df.merge(db[keep], on=['ts_code','trade_date'], how='left')
        print(f"  已合并daily_basic")

    df = DataAdapter.normalize_daily(df)
    stock_count = 0
    for code, grp in df.groupby('ts_code'):
        grp = grp.sort_values('trade_date')
        sp = os.path.join(LOCAL_DATA_DIR, code)
        os.makedirs(sp, exist_ok=True)
        grp.to_csv(os.path.join(sp, 'daily.csv'), index=False, encoding='utf-8-sig')
        stock_count += 1
    print(f"  已保存{stock_count}只股票")

# ====================== D. 验证 ======================
def validate():
    print("\n=== D. 验证 ===")
    sc = sum(1 for d in os.listdir(LOCAL_DATA_DIR)
             if os.path.exists(os.path.join(LOCAL_DATA_DIR, d, 'daily.csv')))
    gf = os.listdir(LOCAL_GLOBAL_DIR)
    print(f"  个股: {sc}只 | 全局文件: {len(gf)}个")
    for f in sorted(gf):
        sz = os.path.getsize(os.path.join(LOCAL_GLOBAL_DIR, f))/1024
        print(f"    {f}: {sz:.0f}KB")

# ====================== MAIN ======================
if __name__ == '__main__':
    s = sys.argv[1] if len(sys.argv) > 1 else '2024-10-01'
    e = sys.argv[2] if len(sys.argv) > 2 else '2024-12-31'

    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    os.makedirs(LOCAL_GLOBAL_DIR, exist_ok=True)

    pro = init()
    fetch_once(pro)
    collectors = fetch_daily(pro, s, e)
    split_daily()
    validate()
    print("\n[完成]")
