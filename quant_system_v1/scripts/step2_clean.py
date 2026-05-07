"""
步骤2：数据清洗
- 硬过滤：ST/停牌/新股
- 质量检测：缺失行/异常值/日期间隔
- 衍生计算：pre_close/涨跌停价格校准
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from config.settings import LOCAL_DATA_DIR, LOCAL_GLOBAL_DIR

LIMIT_RATIOS = {"主板": 0.10, "创业板": 0.20, "科创板": 0.20, "北交所": 0.30}

def detect_market(code):
    sym = code.split('.')[0]
    if sym.startswith(('60','00')): return "主板"
    if sym.startswith('30'): return "创业板"
    if sym.startswith('68'): return "科创板"
    if sym.startswith(('8','4')): return "北交所"
    return "主板"

def main():
    # === 1. 加载元数据 ===
    stock_basic = pd.DataFrame()
    bp = os.path.join(LOCAL_GLOBAL_DIR, 'stock_basic.csv')
    if os.path.exists(bp):
        stock_basic = pd.read_csv(bp, dtype={'ts_code': str})

    # ST列表
    st_codes = set()
    if not stock_basic.empty and 'name' in stock_basic.columns:
        st_mask = stock_basic['name'].str.contains('ST|\\*ST|退', na=False)
        st_codes = set(stock_basic[st_mask]['ts_code'])

    print(f"ST股票: {len(st_codes)}只")

    # === 2. 逐只清洗 ===
    cleaned = 0; skipped_st = 0; skipped_rows = 0; skipped_missing = 0
    total_before = 0; total_after = 0

    for d in os.listdir(LOCAL_DATA_DIR):
        fp = os.path.join(LOCAL_DATA_DIR, d, 'daily.csv')
        if not os.path.exists(fp): continue

        df = pd.read_csv(fp, dtype={'trade_date': str})
        total_before += len(df)

        # 2a. ST过滤
        if d in st_codes:
            os.remove(fp); skipped_st += 1; continue

        # 2b. 最少50个交易日
        if len(df) < 50:
            os.remove(fp); skipped_rows += 1; continue

        # 2c. 必需的OHLCV列
        required = {'open','high','low','close','vol','amount'}
        missing_cols = required - set(df.columns)
        if missing_cols:
            os.remove(fp); skipped_missing += 1; continue

        # 2d. 异常的零值
        for col in ['open','high','low','close']:
            if col in df.columns:
                df = df[df[col] > 0]

        # 2e. 高低价倒挂
        if 'high' in df.columns and 'low' in df.columns:
            df = df[df['high'] >= df['low']]

        # 2f. 按日期排序
        df = df.sort_values('trade_date')

        # 2g. 精确计算 pre_close
        df['pre_close'] = df['close'].shift(1)
        df.loc[df['pre_close'].isna(), 'pre_close'] = df['close']

        # 2h. 精确计算 pct_chg
        df['pct_chg'] = ((df['close'] - df['pre_close']) / df['pre_close'] * 100).round(4)

        # 2i. 涨跌停价格（按板块）
        market = detect_market(d)
        lr = LIMIT_RATIOS.get(market, 0.10)
        df['up_limit'] = (df['pre_close'] * (1 + lr)).round(3)
        df['down_limit'] = (df['pre_close'] * (1 - lr)).round(3)

        # 2j. 标记涨跌停状态
        df['limit_status'] = ''
        df.loc[df['pct_chg'] >= 9.5, 'limit_status'] = 'U'
        df.loc[df['pct_chg'] <= -9.5, 'limit_status'] = 'D'

        # 2k. 连续涨停天数
        df['is_limit_u'] = (df['limit_status'] == 'U').astype(int)
        df['up_down_times'] = df.groupby((df['is_limit_u'] == 0).cumsum())['is_limit_u'].cumsum()

        total_after += len(df)
        df.to_csv(fp, index=False, encoding='utf-8-sig')
        cleaned += 1

    print(f"\n清洗结果:")
    print(f"  有效: {cleaned}只")
    print(f"  剔除ST: {skipped_st}只")
    print(f"  剔除数据不足: {skipped_rows}只")
    print(f"  剔除缺列: {skipped_missing}只")
    print(f"  行数: {total_before} → {total_after} ({total_before-total_after}条异常)")

if __name__ == '__main__':
    main()
