# 量化交易系统 — 项目上下文

## 项目概况
- 量化交易系统 v1，A股短线策略（2-3天持股周期）
- 盈亏比 1:2，最长持仓 3 天
- 第三方 Tushare 接口: token=`123396cf48bacd87370d4541fe4c2c51bcacb43f32f66890650dd5bb907a`, api=`http://jiaoch.site`
- Python 路径: `C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe`

## 目录结构
```
F:\编程文件\
├── quant_system_v1/          # 主项目（模块化架构）
│   ├── main.py               # CLI入口: fetch/backtest/pick/monitor
│   ├── config/               # settings.py + strategy_config.py
│   ├── data_source/          # 多源降级: Tushare→AKShare→东财→Baostock
│   ├── backtest/engine.py    # 事件驱动回测
│   ├── strategy/             # 3个策略: 打板/缩量潜伏/板块轮动
│   ├── factor_lib/           # 40+因子（技术/基本面/价量/情绪/板块宏观）
│   ├── stock_picker/         # 盘后选股
│   ├── monitor/realtime.py   # 实时监控（需要重写）
│   ├── scripts/              # step1_fetch.py, step2_clean.py, fetch_all.py
│   └── utils/
├── Untitled-12_comprehensive_optimized.py  # 原始单文件（已被quant_system_v1替代）
├── data/                     # 全局CSV数据
├── data_all_stocks/          # 按股票拆分的日线数据
├── tushare_api_reference.md  # Tushare 231接口文档
└── 使用教程/
```

## 已完成
- [x] 模块化重构（从Untitled-12拆分）
- [x] Tushare连接验证: stock_basic(5499只), daily(5460条/日)
- [x] 数据抓取脚本: step1_fetch.py + step2_clean.py + fetch_all.py（15+接口）
- [x] 事件驱动回测引擎
- [x] 3策略 + 评分规则
- [x] 因子注册系统
- [x] 多数据源降级
- [x] 市场状态检测 + 统一策略管线
- [x] Superpowers 已安装（14技能），settings.json路径已修复

## 待完成（双线并行）
### 线1：数据 + 回测
- [ ] 执行全量数据抓取（先跑2024-10-01到2024-12-31快速验证）
- [ ] 向量化回测引擎（参数扫描/因子IC分析）
- [ ] 因子评价体系（IC/IR/分组回测/去极值/中性化/标准化）
- [ ] 因子合成（等权/IC加权/最优化）
- [ ] 回测增加真实涨跌停限制

### 线2：策略 + 监控
- [ ] 新增策略：首板策略/龙头反包/断板反包
- [ ] 实时监控重写（自动获取监控列表+信号生成）
- [ ] 企微推送（Webhook已配置）
- [ ] 盘前集合竞价强度分析
- [ ] 板块热度/资金流向监控
- [ ] 所有策略迁移到统一管线(pipeline.py)

### 后续
- [ ] 参数优化（网格搜索/贝叶斯优化）
- [ ] 多策略组合回测
- [ ] HTML回测报告
- [ ] ETF/可转债支持

## 三板代码映射
- 主板: 60xxxx/00xxxx (涨跌停±10%)
- 创业板: 30xxxx (±20%)
- 科创板: 68xxxx (±20%)
- 北交所: 8xxxxx/4xxxxx (±30%)
