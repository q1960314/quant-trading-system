# 量化交易系统 v3 — 自主进化设计规格

## 概述

在 v2 基础上（7策略/66因子/双引擎/分析层），新增自主进化能力：公式挖掘自动生成新因子、数据源扫描发现未用接口、因子进化自适应市场、GitHub策略搜索+回测对比、ML涨跌预测、持续学习调度。系统从"手工优化"升级为"自主进化"。

## 架构

```
quant_system_v1/
├── evolution/                        # 【新增】进化模块
│   ├── formula_miner.py              # 遗传编程因子公式挖掘
│   ├── datasource_scanner.py         # Tushare接口自动扫描+因子注册
│   ├── factor_evolution.py           # 因子轮动+权重自适应+生命周期
│   └── github_strategy_miner.py      # GitHub搜索→提取策略→回测对比
├── ml/                               # 【新增】机器学习
│   ├── predictor.py                  # LightGBM涨跌预测
│   ├── feature_engineering.py        # 自动特征工程(交叉/滞后/聚合)
│   └── model_manager.py              # 模型训练/版本管理/重训练
├── scheduler/                        # 【新增】调度
│   └── scheduler.py                  # 定时任务(日/周/月) + 手动触发
├── evolution_output/                  # 【新增】产出目录
│   ├── factors/                      # 挖掘出的新因子
│   ├── strategies/                   # GitHub提取的策略
│   └── models/                       # ML模型文件
└── [v2已有模块不变]
```

新增3个模块，8个文件。

---

## 模块详情

### 1. 公式挖掘器 (evolution/formula_miner.py)

**基因编码**: 因子公式 = 树结构, 节点=(操作符, [子节点])
- 操作符: add, sub, mul, div, safe_div(a,b+eps), rolling_mean(n), rolling_std(n), rank, delay(n), delta(n), ts_corr(a,b,n), signed_power
- 叶子: open, high, low, close, vol, amount, turnover_rate, pre_close
- 约束: 最大深度4, 自动防除零, 至少包含1个价格项

**遗传算法参数**: 种群100, 20代, 交叉0.4, 变异0.3, 精英10
- 初始化: 50随机 + 50基于现有因子变异
- 适应度: |ICIR| (训练集2022-2023, 验证集2024)
- 通过: |ICIR|>0.3 且 Top/Bottom分层单调
- 输出: 通过公式 → 生成FactorBase子类代码 → 追加到factor_lib/

### 2. 数据源扫描器 (evolution/datasource_scanner.py)

- 解析 tushare_api_reference.md (231接口) → 提取接口名/参数/返回列
- 对比已有CSV表 → 发现未使用接口
- 对未用接口: 测试调用(2024-01-01~2024-01-10小范围) → 检查数据覆盖率
- 每个新数值列 → 自动生成因子类 → 追加到factor_lib/
- 标记: AVAILABLE / NO_PERMISSION / EMPTY_DATA

### 3. 因子进化器 (evolution/factor_evolution.py)

- **FactorRotator**: 按BULL/BEAR/NORMAL分组计算因子ICIR, 牛市优先动量, 熊市优先低波+质量
- **AdaptiveWeightOptimizer**: 每月重算IC, 用6月ICIR加权, ICIR连续2月<0.15标记退火
- **FactorLifecycleMonitor**: 日记录IC值, 检测趋势(上升/稳定/下降/崩溃), 预警IC下降>30%

### 4. GitHub策略挖掘器 (evolution/github_strategy_miner.py)

- gh search repos关键词: "A股 量化策略" "涨停策略" "均线突破" "龙头战法" "短线回测"
- 筛选: stars>5, created>=2024, 语言Python, Top 20
- clone到sandbox/ → AST解析提取策略类 → StrategyConverter包装为StrategyBase
- StrategyComparator: 统一回测2024年, 与现有7策略对比排名, Top 3注册

### 5. ML预测器 (ml/predictor.py)

- 特征: 66因子 + 交叉特征(f_i×f_j, top 50 by IC) + 滞后特征(t-1,t-2,t-5) + 行业均值
- 模型: LightGBM (objective=binary, metric=auc, early_stopping=50)
- 训练: 2022-2023训练, 2024验证
- 输出: 涨跌概率[0,1] → 注册为第8个策略 "ML预测策略"
- 重训练: 每月1号, 保留最近3个模型版本, AUC提升才上线

### 6. 调度器 (scheduler/scheduler.py)

- 每日09:00: 更新因子IC记录; 15:30: 增量数据抓取
- 每周: 因子生命周期检查; 策略滚动回测验证
- 每月: ML重训练; GitHub策略搜索; 数据源扫描; 公式挖掘
- CLI: `python main.py evolve --mode factors|strategies|ml|full`

---

## 数据流

```
Tushare API 231接口
  → datasource_scanner → 新列 → 注册因子
price/vol数据
  → formula_miner → 遗传编程 → ICIR筛选 → 注册因子
GitHub搜索
  → clone → AST提取 → 回测对比 → 注册策略
因子+特征
  → LightGBM训练 → 涨跌概率 → 第8策略
factor_evolution → 因子轮动/权重/淘汰
scheduler → 定时触发以上流程
```

---

## 兼容性

- v1/v2 代码不改动，evolution/和ml/为独立新模块
- 公式挖掘产出为标准FactorBase子类，直接追加到factor_lib/
- 策略挖掘产出为标准StrategyBase子类，直接追加到strategy/
- scheduler为独立脚本，可单独运行或通过cron/CLI触发
- 所有产出写入evolution_output/目录

---

## 风险

- 遗传编程公式可能过拟合训练集，需验证集过滤
- GitHub策略代码质量参差不齐，需人工审核标记
- ML模型在极端行情下可能失效，需结合止损
- Tushare部分接口无权限，scanner需处理权限错误
