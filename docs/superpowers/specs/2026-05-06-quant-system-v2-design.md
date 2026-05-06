# 量化交易系统 v2 — 设计规格

## 概述

在 v1 模块化架构基础上，深化回测体系、建立因子评价系统、引入参数优化、替换数据存储为 DuckDB、补齐真实交易约束和策略组合管理。v1 的 3 个策略和数据抓取管线保持不动，增量叠加能力。

## 架构总览

```
quant_system_v1/
├── main.py                          # CLI入口（不变）
├── config/
│   ├── settings.py                  # [扩展] 优化/因子评价/DB配置
│   └── strategy_config.py           # [扩展] 每个策略独立参数空间
├── data_source/                     # 数据源层（已有，微调）
│   ├── tushare_source.py            # 保留
│   ├── akshare_source.py            # 保留
│   ├── eastmoney_source.py          # 保留
│   ├── baostock_source.py           # 保留
│   ├── fetcher.py                   # [扩展] 并发动态检测+断点续传
│   └── adapter.py                   # [扩展] 统一OHLCV+复权+停牌标记
├── data_warehouse/                  # 【新增】数据仓库
│   ├── cleaner.py                   # 缺失值/异常值/复权校准
│   ├── validator.py                 # 完整性校验(交易日历对齐)
│   ├── version.py                   # 数据快照版本管理
│   └── duckdb_store.py              # DuckDB列式存储+SQL查询
├── backtest/
│   ├── engine.py                    # 现有事件驱动引擎(保留)
│   ├── vectorized_engine.py         # 【新增】向量化引擎
│   ├── constraints.py               # 【新增】真实约束系统
│   ├── portfolio.py                 # 【新增】多策略组合层
│   └── report.py                    # 【新增】HTML/Plotly报告
├── factor_lib/
│   ├── registry.py                  # 已有
│   ├── price_factors.py             # [扩展] 振幅/集合竞价/分时形态
│   ├── technical_factors.py         # [扩展] 布林/ATR/ADX
│   ├── fundamental_factors.py       # [扩展] EP_TTM/PEG/ROE变动
│   ├── sentiment_factors.py         # [扩展] 游资参与/龙虎榜/热榜
│   ├── sector_macro_factors.py      # [扩展] 板块RPS/资金流向/CPI-PMI
│   ├── evaluation.py                # 【新增】因子评价(IC/分层/衰减)
│   ├── neutralization.py            # 【新增】行业/市值/Barra中性化
│   └── synthesis.py                 # 【新增】因子合成
├── optimizer/                       # 【新增】参数优化
│   ├── grid_search.py               # 网格搜索(粗筛)
│   ├── walk_forward.py              # Walk-Forward(防过拟合)
│   └── bayesian.py                  # 贝叶斯优化(Optuna精调)
├── analysis/                        # 【新增】分析层
│   ├── transaction_cost.py          # 真实成本模型
│   ├── attribution.py               # Brinson+因子归因
│   ├── correlation.py               # 策略相关性矩阵
│   ├── competition.py               # 多策略竞争排名
│   ├── alpha_decay.py               # Alpha衰减曲线
│   └── stress_test.py               # 压力测试
├── strategy/                        # 策略层(已有+迁移)
│   ├── base.py                      # 已有
│   ├── pipeline.py                  # [扩展] 集成因子评价反馈
│   ├── board_strategy.py            # 已有
│   ├── shrink_volume.py             # 已有
│   ├── sector_rotation.py           # 已有
│   └── ma_breakout_strategy.py      # [迁移] 迁入统一管线
├── stock_picker/                    # 选股(已有)
├── monitor/                         # 监控(B阶段)
├── scripts/                         # 已有
└── utils/                           # 已有
```

新增 4 个模块: data_warehouse, optimizer, analysis; 扩展 3 个模块: backtest, factor_lib, strategy。共约 20 个新文件。

---

## 模块详情

### 1. 向量化回测引擎 (vectorized_engine.py)

将逐行遍历改为 pandas/numpy 矩阵运算，性能提升 50-100x。

**核心数据结构**:
- `price_matrix`: (T, N) 收盘价矩阵, T=交易日数, N=股票数
- `signal_matrix`: (T, N) 信号矩阵, 1=buy, -1=sell, 0=hold
- `constraint_matrix`: (T, N) bool, True=可交易
- `cost_matrix`: (T, N) 交易成本率, 预计算
- `position_matrix`: (T, N) 持仓权重

**计算流程**:
1. pivot原始DataFrame为矩阵
2. numpy批量计算因子+信号
3. 单层循环逐日撮合(无内层股票循环)
4. 返回BacktestResult

**策略适配接口**:
```python
class StrategyBase(ABC):
    def generate_signals_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """返回(T,N)信号矩阵"""
        pass
```

### 2. 真实约束系统 (constraints.py)

四个约束类, 统一接口 `apply(df) -> bool_mask`:

- **LimitUpDownConstraint**: 涨跌停当日不可买入(主板±10%, 创业板/科创±20%, 北交所±30%)
- **SuspensionConstraint**: 停牌日过滤, 成交量=0或金额=0标记为不可交易
- **LiquidityConstraint**: 最小成交额(默认10万)+最小换手率(默认3%)
- **PositionConstraint**: 单票20%上限, 单行业30%上限, 总持仓≤5只

涨跌停排队概率模型:
- 涨停封单量/流通盘 < 0.5%: 买到概率 50%
- 涨停封单量/流通盘 0.5-1%: 买到概率 20%
- 涨停封单量/流通盘 > 1%: 买到概率 5%

### 3. 真实交易成本 (analysis/transaction_cost.py)

```
买入成本 = 佣金(0.025%) + 滑点 + 冲击成本
卖出成本 = 佣金(0.025%) + 印花税(0.1%) + 滑点 + 冲击成本
冲击成本 = f(买入金额/日均成交额), 基准0.1%, 最大1%
不同板块费率: 沪市过户费0.001%, 深市0, 创业板0
```

### 4. 因子评价 (factor_lib/evaluation.py)

**IC分析**:
- IC = corr(factor[t], forward_return[t+1])
- ICIR = mean(IC) / std(IC), 阈值>0.3通过
- IC胜率 = IC>0的比例
- IC衰减: 滞后1/3/5/10/20日的IC曲线

**分层回测**:
- 按因子值分5组, 等权持有, 日度再平衡
- 输出: Top组/Bottom组年化、多空收益、单调性检验

**评价报告字段**: 因子名, 类型, IC均值, ICIR, IC胜率, Top年化, Bottom年化, 多空夏普, 中性化后IC留存率, 评价日期, 是否通过

### 5. 因子中性化 (factor_lib/neutralization.py)

- 行业中性化: factor = factor - industry_mean
- 市值中性化: 对ln(market_cap)回归取残差
- Barra CNE5正交: 对市场/规模/价值/盈利/投资5因子正交

### 6. 因子合成 (factor_lib/synthesis.py)

- 等权: composite = mean(f1,f2,...,fn)
- IC加权: weight_i = |IC_i| / sum(|IC|)
- ICIR最大化: 优化权重向量使复合因子ICIR最大
- 逐步回归: stepwise forward selection

### 7. 参数优化 (optimizer/)

三层递进:
1. **网格搜索**: 粗筛Top 10参数组合(夏普最高)
2. **Walk-Forward**: 训练12月→验证3月→滚动3月→重复。评估样本外夏普稳定性
3. **贝叶斯优化**(Optuna TPE): 在最优±30%邻域精调, 30-50次迭代

### 8. 策略组合管理 (backtest/portfolio.py)

- **资金分配**: 等权/Kelly/RiskParity三种分配器
- **相关矩阵**: 计算各策略日收益相关性, 高相关(>0.5)时降低同步配置
- **波动率目标**: 目标10%年化波动率, 动态调整仓位

### 9. 业绩归因 (analysis/attribution.py)

**Brinson归因**: 总收益=基准+配置贡献+选股贡献+交互项
**因子归因**: Fama-French五因子拆解, 如果α显著>0说明策略有真Alpha

### 10. Alpha衰减 (analysis/alpha_decay.py)

计算信号不同延迟的收益衰减:
- T日收盘信号 → T+1开盘买入: 基准(衰减0%)
- T日收盘信号 → T+1收盘买入: 预期衰减~15%
- T+2开盘买入: ~35%, T+2收盘: ~50%

### 11. 压力测试 (analysis/stress_test.py)

5个历史场景 + 自定义:
- 2015-06~09(股灾,跌45%), 2016-01熔断, 2018全年熊市
- 2020-02-03(疫情,3000+跌停), 2024-02微盘股危机
- 自定义: 单日跌7%, 连跌3日3%

输出每个场景: 最大回撤/回本天数/策略止损触发情况

### 12. 多策略竞争排名 (analysis/competition.py)

各策略候选股 → 标准化得分→合并排名:
- 多策略共振股票(≥2个策略推荐): 额外加分
- 最终输出 Top 20 合并候选池

### 13. DuckDB数据仓库 (data_warehouse/duckdb_store.py)

单文件 `quant.duckdb`, 列式存储, 20+张表对应现有CSV。
回测加载从"扫描5000个CSV文件"改为"一条SQL聚合查询"。
CSV→DuckDB迁移脚本一次性执行, 原CSV保留为备份。

---

## 数据流

```
Tushare/东财/AKShare
        │
        ▼
   data_source/fetcher.py  ← 并发抓取+断点续传
        │
        ▼
   data_warehouse/cleaner.py  ← 清洗/复权/去重
        │
        ▼
   data_warehouse/duckdb_store.py  ← DuckDB存储
        │
        ├──→ backtest/vectorized_engine.py  ← 回测
        │       │
        │       ├──→ analysis/transaction_cost.py
        │       ├──→ analysis/attribution.py
        │       ├──→ analysis/stress_test.py
        │       └──→ backtest/report.py  ← HTML报告
        │
        ├──→ factor_lib/evaluation.py  ← 因子评价
        │       │
        │       ├──→ neutralization.py
        │       └──→ synthesis.py
        │
        ├──→ optimizer/  ← 参数优化
        │
        └──→ stock_picker/  ← 选股
```

---

## 兼容性

- v1 事件驱动引擎保留, 通过 `--engine vectorized` 切换
- v1 的 3 个策略 filter()/score() 接口保留, 新增 generate_signals_vectorized() 可选
- 配置向后兼容, 新增配置项有默认值
- 原CSV数据不删除, DuckDB为增量添加

---

## 执行阶段

| 阶段 | 内容 | 新文件数 |
|------|------|----------|
| A1 | 向量化回测+真实约束 | 3 |
| A2 | 真实成本模型 | 1 |
| A3 | 因子评价+中性化+合成+Alpha衰减 | 5 |
| A4 | 参数优化(Walk-Forward+贝叶斯) | 3 |
| A5 | DuckDB数仓+数据迁移 | 3 |
| A6 | 策略组合+相关性+归因+竞争排名 | 4 |
| A7 | 压力测试 | 1 |
| A8 | HTML报告+全量验证 | 1 |

---

## 风险与约束

- Tushare第三方接口(jiaoch.site)并发上限=2, 不可突破
- 小市值股票(北交所)涨跌停30%, 回测需区分板块
- 2020年起的数据, 部分2018-2019接口可能无数据
- DuckDB需要Python 3.8+, 当前环境Python 3.11满足
- Optuna需要pip安装
