"""策略专属配置 + 评分规则"""
STRATEGY_CONFIG = {
    "打板策略": {
        "type": "link", "min_order_ratio": 0.03, "max_break_times": 1,
        "link_board_range": [2, 4], "exclude_late_board": True,
        "stop_loss_rate": 0.06, "stop_profit_rate": 0.12, "max_hold_days": 2,
    },
    "缩量潜伏策略": {
        "type": "first_board_pullback", "first_board_limit": True,
        "board_volume_growth": 1.5, "shrink_volume_ratio": [1/3, 1/2],
        "shrink_days_range": [3, 10], "pullback_support_level": 0.5,
        "support_tolerance": 0.02, "stop_loss_rate": 0.03,
        "stop_profit_rate": 0.15, "max_hold_days": 8, "rotate_days": 1,
    },
    "板块轮动策略": {
        "type": "industry", "rotate_days": 3, "stop_loss_rate": 0.05,
        "stop_profit_rate": 0.1, "main_trend": True, "fund_inflow_top": 30, "max_hold_days": 3,
    },
}

SCORING_RULES = {
    "pass_score_default": 12, "enable_dynamic_weight": True,
    "strategy_pass_score": {"打板策略": 10, "缩量潜伏策略": 12, "板块轮动策略": 15},
    "items": {
        "缩量到首板1/3以内": [3, "current_vol_ratio <= 0.33", {"打板策略":0,"缩量潜伏策略":3,"板块轮动策略":0}],
        "缩量到首板1/3~1/2": [2, "current_vol_ratio > 0.33 and current_vol_ratio <= 0.5", {"打板策略":0,"缩量潜伏策略":2,"板块轮动策略":0}],
        "精准回踩支撑位±1%": [3, "abs(price_to_support_ratio) <= 0.01", {"打板策略":0,"缩量潜伏策略":3,"板块轮动策略":0}],
        "回踩支撑位±2%": [2, "abs(price_to_support_ratio) > 0.01 and abs(price_to_support_ratio) <= 0.02", {"打板策略":0,"缩量潜伏策略":2,"板块轮动策略":0}],
        "首板放量≥2倍": [2, "board_vol_growth >= 2", {"打板策略":0,"缩量潜伏策略":2,"板块轮动策略":0}],
        "首板放量≥1.5倍": [1, "board_vol_growth >= 1.5 and board_vol_growth < 2", {"打板策略":0,"缩量潜伏策略":1,"板块轮动策略":0}],
        "回调天数3-5天": [2, "days_after_board >=3 and days_after_board <=5", {"打板策略":0,"缩量潜伏策略":2,"板块轮动策略":0}],
        "回调天数6-10天": [1, "days_after_board >=6 and days_after_board <=10", {"打板策略":0,"缩量潜伏策略":1,"板块轮动策略":0}],
        "流通市值50亿-200亿": [1, "float_market_cap >= 50 and float_market_cap <= 200", {"打板策略":1,"缩量潜伏策略":1,"板块轮动策略":1}],
        "成交额≥1亿": [1, "amount >= 100000", {"打板策略":1,"缩量潜伏策略":1,"板块轮动策略":1}],
        "无减持公告": [1, "no_reduction == 1", {"打板策略":1,"缩量潜伏策略":1,"板块轮动策略":1}],
        "无监管问询": [1, "no_inquiry == 1", {"打板策略":1,"缩量潜伏策略":1,"板块轮动策略":1}],
        "连板高度≥3板": [2, "up_down_times >= 3", {"打板策略":2,"缩量潜伏策略":0,"板块轮动策略":0.5}],
        "连板高度2板": [1, "up_down_times == 2", {"打板策略":2,"缩量潜伏策略":0,"板块轮动策略":0.5}],
        "机构净买入≥5000万": [2, "inst_buy >= 5000", {"打板策略":2,"缩量潜伏策略":0,"板块轮动策略":1.5}],
        "游资净买入≥3000万": [2, "youzi_buy >= 3000", {"打板策略":2,"缩量潜伏策略":0,"板块轮动策略":1}],
        "主线行业匹配": [2, "is_main_industry == 1", {"打板策略":1,"缩量潜伏策略":0,"板块轮动策略":2}],
        "热点题材≥2个": [2, "concept_count >= 2", {"打板策略":1,"缩量潜伏策略":0,"板块轮动策略":2}],
        "换手率3%-10%": [1, "turnover_ratio >= 3 and turnover_ratio <= 10", {"打板策略":2,"缩量潜伏策略":0,"板块轮动策略":1}],
        "封单金额≥1亿": [1, "order_amount >= 10000", {"打板策略":2,"缩量潜伏策略":0,"板块轮动策略":0}],
        "非尾盘封板": [1, "first_limit_time <= '14:30'", {"打板策略":2,"缩量潜伏策略":0,"板块轮动策略":0}],
        "炸板次数≤3次": [1, "break_limit_times <= 3", {"打板策略":2,"缩量潜伏策略":0,"板块轮动策略":0}],
    }
}
