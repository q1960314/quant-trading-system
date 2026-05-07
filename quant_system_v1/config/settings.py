"""
量化交易系统 v1 — 统一配置中心
"""
import os

# ========== 运行模式 ==========
AUTO_RUN_MODE = "每日选股"
STRATEGY_TYPE = "打板策略"
ALLOWED_MARKET = ["主板"]

# ========== 时间 ==========
START_DATE = "2020-01-01"
END_DATE = "2026-05-06"

# ========== 数据源 ==========
DATA_SOURCE_PRIORITY = ["tushare", "akshare", "eastmoney", "baostock"]
TUSHARE_TOKEN = "123396cf48bacd87370d4541fe4c2c51bcacb43f32f66890650dd5bb907a"
TUSHARE_API_URL = "http://jiaoch.site"

# ========== 交易参数 ==========
INIT_CAPITAL = 5000
MAX_HOLD_DAYS = 3
STOP_LOSS_RATE = 0.06
STOP_PROFIT_RATE = 0.12
COMMISSION_RATE = 0.00025
MIN_COMMISSION = 5
STAMP_TAX_RATE = 0.001
SINGLE_STOCK_POSITION = 0.2
INDUSTRY_POSITION = 0.3
MAX_HOLD_STOCKS = 5
MAX_DRAWDOWN_STOP = 0.15
DRAWDOWN_STOP_DAYS = 3
MAX_TRADE_RATIO = 0.05
PRICE_ADJUST = "front"
SLIPPAGE_RATE = 0.005
STANDALONE_MODE = True

# ========== 抓取 ==========
FETCH_OPTIMIZATION = {
    'max_workers': 2,  # jiaoch.site API并发上限=2
    'batch_io_interval': 10,
    'max_requests_per_minute': 120
}
FILTER_CONFIG = {"min_amount": 100000, "min_turnover": 3, "exclude_st": True, "max_fetch_retry": 3}

# ========== 选股 ==========
STOCK_PICK_CONFIG = {"min_pick_score": 16, "max_output_count": 10, "export_excel": True, "fetch_days": 1}

# ========== 路径 ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DATA_DIR = os.path.join(BASE_DIR, '..', 'data_all_stocks')
LOCAL_GLOBAL_DIR = os.path.join(BASE_DIR, '..', 'data')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
for d in [LOCAL_DATA_DIR, LOCAL_GLOBAL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ========== 微信 ==========
WECHAT_ROBOT_ENABLED = False
WECHAT_ROBOT_URL = ""

# ========== 涨跌停比例 ==========
LIMIT_RATIO_MAP = {"主板": 0.10, "创业板": 0.20, "科创板": 0.20, "北交所": 0.30}

LOG_LEVEL = "INFO"

# ========== 向量化回测 ==========
VECTORIZED_ENGINE_ENABLED = True
CONSTRAINT_CONFIG = {
    'min_amount': 100000, 'min_turnover': 3.0,
    'exclude_st': True, 'exclude_suspend': True,
    'limit_up_buyable': False, 'max_position_stocks': 5,
    'single_stock_ratio': 0.2, 'industry_ratio': 0.3,
    'max_daily_trade_ratio': 0.05,
}

# ========== 因子评价 ==========
FACTOR_EVAL_CONFIG = {
    'min_obs': 100, 'icir_threshold': 0.3,
    'n_groups': 5, 'decay_lags': [1, 3, 5, 10, 20],
}

# ========== 参数优化 ==========
OPTIMIZER_CONFIG = {
    'train_months': 12, 'valid_months': 3,
    'n_trials_bayesian': 50,
}

# ========== DuckDB ==========
DUCKDB_PATH = None

# ========== 压力测试 ==========
STRESS_SCENARIOS = [
    ('2015-06-12', '2015-09-15'),
    ('2016-01-04', '2016-01-07'),
    ('2018-01-01', '2018-12-31'),
    ('2020-02-03', '2020-02-03'),
    ('2024-01-15', '2024-02-07'),
]
