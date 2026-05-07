# 量化交易系统 v4 — 项目上下文

## 启动
```bash
cd F:\编程文件\quant_system_v1
py main.py fetch --end 2026-05-07    # 抓数据(重启后第一件事)
py main.py backtest --strategy 打板策略
py scripts/run_all_backtests.py
py scripts/pipeline.py
```

## 关键配置
- Token: 123396cf48bacd87370d4541fe4c2c51bcacb43f32f66890650dd5bb907a
- API: http://jiaoch.site
- Python: C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe
- 限频: 120/min, 并发2

## 架构
```
quant_system_v1/
├── main.py          # CLI: fetch/backtest/pick/monitor/evolve
├── backtest/        # vectorized_engine(默认) + engine(事件驱动)
├── strategy/        # 8策略
├── factor_lib/      # 66因子 + 评价/中性化/合成
├── data_source/     # fetcher(含并发限频) + concurrency(RateLimiter)
├── ml/              # LightGBM预测器
├── evolution/       # 遗传编程/扫描器/因子进化/GitHub挖掘
├── optimizer/       # 网格/Walk-Forward/贝叶斯
├── analysis/        # 成本/归因/相关性/压力/竞争排名
├── data_warehouse/  # DuckDB存储+清洗
├── scheduler/       # 定时调度
└── scripts/         # pipeline/benchmark/run_all_backtests等
```

## 已知问题
- 数据仅102天, 需全量抓取(2020-2026)
- DuckDB文件>100MB, gitignore已排除
- 5个策略参数需调优(板块/龙头/均线/缩量/首板)
- 扫描器需API名称映射(Tushare中文名→英文方法名)
