"""通用辅助函数"""

def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def ts_code_to_symbol(code):
    """000001.SZ -> 000001"""
    if not code:
        return ""
    return str(code).split('.')[0]

def detect_market(code):
    """根据代码判断板块: 主板/创业板/科创板/北交所"""
    sym = ts_code_to_symbol(code)
    if sym.startswith(('60', '00')):
        return "主板"
    elif sym.startswith('30'):
        return "创业板"
    elif sym.startswith('68'):
        return "科创板"
    elif sym.startswith(('8', '4')):
        return "北交所"
    return "主板"

def get_limit_ratio(code):
    """获取涨停比例"""
    m = detect_market(code)
    ratios = {"主板": 0.10, "创业板": 0.20, "科创板": 0.20, "北交所": 0.30}
    return ratios.get(m, 0.10)

def fmt_date(d, fmt="%Y%m%d"):
    from datetime import datetime, date
    if isinstance(d, (date, datetime)):
        return d.strftime(fmt)
    return str(d)
