# 量化交易系统 v4 — 生产就绪设计规格

## 概述

系统已有骨架（v1-v3，8策略/66因子/双引擎/进化/ML），但多个模块未实际运行。v4 目标：每一个模块都跑通、数据拉满、速度优化、一键自动化。

## 架构变更

```
变动点:
1. backtest/engine.py → 增加 DuckDB 加载路径 (use_duckdb=True)
2. scripts/pipeline.py → 新增：端到端自动化脚本
3. 数据层 → scanner 发现的新接口自动追加抓取
```

## 模块详情

### 1. 数据源扫描 + 新接口抓取

- 运行 TushareAPIScanner 扫描剩余 89 个未使用接口
- 对 AVAILABLE 接口：自动测试抓取（小日期范围验证）
- 将可用新接口加入抓取管线
- 目标：从 26 个扩展到 40+ 个接口

### 2. DuckDB 回测集成

- vectorized_engine.load_data() 增加 DuckDB 模式
- SQL 替代 CSV 扫描：数据加载从 8s → <1s
- 全局数据合并通过 SQL JOIN 完成
- CSV 模式保留作为 fallback

### 3. 完整历史数据拉取

- `python main.py fetch --start 2020-01-01 --end 2024-09-30`
- 拉取完成后导入 DuckDB
- 日线数据从 61 天扩展到 ~1200 天

### 4. Walk-Forward 参数优化实跑

- 在打板策略上运行 3 轮 Walk-Forward
- 输出最优参数 + 样本外夏普稳定性曲线
- 模型保存并可用于选股

### 5. 端到端自动化脚本

```
python scripts/pipeline.py
  1. 增量数据抓取
  2. DuckDB 更新
  3. 因子 IC 刷新
  4. 8 策略回测
  5. 多策略选股
  6. HTML 报告
```

### 6. 性能基准

| 指标 | 当前 | 目标 |
|------|------|------|
| 数据加载 | 8s (5350 CSV) | <1s (DuckDB) |
| 8策略回测 | ~100s | ~30s |
| 因子IC分析 | ~180s | ~60s |

## 数据流

```
Tushare API (115接口)
  → scanner → 发现可用新接口 → 追加抓取
  → fetcher → CSV → DuckDB
  → backtest ← DuckDB SQL ← 全局数据 JOIN
  → optimizer → Walk-Forward → 最优参数
  → pipeline.py → 一键自动化
```

## 兼容性

- 现有 CSV 加载路径保留（--engine event 使用 CSV）
- 向量化引擎默认使用 DuckDB
- 新接口抓取追加到现有 CSV 目录
