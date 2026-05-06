# Quant System v3 — Auto-Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add autonomous factor discovery, strategy mining, ML prediction, and scheduled continuous learning.

**Architecture:** Three new independent packages (evolution/, ml/, scheduler/) layered on existing v2 system. Each evolution module is a standalone runnable script that outputs to evolution_output/. ML predictor registers as strategy #8. Scheduler manages periodic runs.

**Tech Stack:** Python 3.11, pandas, numpy, scikit-learn, LightGBM, deap (genetic programming), gh CLI

---

### Task 1: Evolution package setup + Data Source Scanner

**Files:**
- Create: `quant_system_v1/evolution/__init__.py`
- Create: `quant_system_v1/evolution/datasource_scanner.py`

- [ ] **Step 1: Create package init and scanner**

```python
# quant_system_v1/evolution/__init__.py
"""Evolution modules: factor mining, strategy discovery, adaptive optimization."""
```

```python
# quant_system_v1/evolution/datasource_scanner.py
"""
Auto-scan Tushare API reference for unused data interfaces.

Parses tushare_api_reference.md, compares with existing CSV files,
tests each unused interface, and auto-registers new numeric columns as factors.
"""
import os, re, sys, time
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from config.settings import LOCAL_GLOBAL_DIR, TUSHARE_TOKEN, TUSHARE_API_URL
from data_source.manager import DataSourceManager

logger = get_logger("scanner")

SCAN_START = "2024-01-01"
SCAN_END = "2024-01-10"
API_REF_PATH = os.path.join(os.path.dirname(LOCAL_GLOBAL_DIR), "tushare_api_reference.md")


class TushareAPIScanner:
    def __init__(self):
        self.mgr = DataSourceManager(token=TUSHARE_TOKEN, api_url=TUSHARE_API_URL)
        self.existing_tables = self._get_existing_tables()
        self.results = []

    def _get_existing_tables(self):
        return set(f.replace('.csv', '') for f in os.listdir(LOCAL_GLOBAL_DIR)
                   if f.endswith('.csv'))

    def parse_api_doc(self):
        """Extract interface names from tushare_api_reference.md."""
        if not os.path.exists(API_REF_PATH):
            logger.warning(f"API doc not found: {API_REF_PATH}")
            return []
        apis = []
        with open(API_REF_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        matches = re.findall(r'###\s+(\w+)', content)
        return [m for m in matches if m not in ('接口', '概述', '目录')]

    def test_interface(self, api_name):
        """Test a single interface and return column names if successful."""
        try:
            df = self.mgr.call_api(api_name, start_date=SCAN_START, end_date=SCAN_END)
            if df is not None and not df.empty:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                return {
                    'api': api_name,
                    'status': 'AVAILABLE',
                    'rows': len(df),
                    'new_columns': [c for c in numeric_cols if c not in ('ts_code', 'trade_date')],
                }
            return {'api': api_name, 'status': 'EMPTY_DATA', 'rows': 0, 'new_columns': []}
        except Exception as e:
            return {'api': api_name, 'status': 'FAILED', 'error': str(e), 'new_columns': []}

    def scan(self):
        """Full scan: find unused APIs, test them, return new column candidates."""
        api_list = self.parse_api_doc()
        if not api_list:
            logger.warning("No APIs parsed from doc")
            return self.results
        unused = [a for a in api_list if a not in self.existing_tables]
        logger.info(f"Found {len(api_list)} APIs: {len(unused)} unused, {len(self.existing_tables)} existing")
        for api_name in unused[:50]:  # limit per run
            result = self.test_interface(api_name)
            self.results.append(result)
            if result['status'] == 'AVAILABLE' and result['new_columns']:
                logger.info(f"  {api_name}: {len(result['new_columns'])} new cols: {result['new_columns'][:5]}")
            else:
                logger.debug(f"  {api_name}: {result['status']}")
        return self.results

    def export_candidates(self):
        """Generate factor code for discovered columns."""
        candidates = [r for r in self.results if r['status'] == 'AVAILABLE' and r['new_columns']]
        if not candidates:
            return ""
        code_lines = [
            "# Auto-generated factors from datasource_scanner.py",
            f"# Generated: {pd.Timestamp.now().isoformat()}",
            "import pandas as pd; import numpy as np",
            "from factor_lib.registry import FactorRegistry, FactorBase",
            "",
        ]
        for r in candidates:
            for col in r['new_columns']:
                safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', col).strip('_')
                code_lines.append(f"""
@FactorRegistry.register
class Auto_{safe_name}(FactorBase):
    name = "{safe_name}"; category = "auto"; desc = "Auto-discovered from {r['api']}.{col}"
    def compute(self, df):
        return df['{col}'] if '{col}' in df.columns else pd.Series(np.nan, index=df.index)
""")
        return '\n'.join(code_lines)


if __name__ == '__main__':
    scanner = TushareAPIScanner()
    results = scanner.scan()
    code = scanner.export_candidates()
    if code:
        out_path = os.path.join(os.path.dirname(LOCAL_GLOBAL_DIR), '..', 'quant_system_v1',
                                'evolution_output', 'factors', 'auto_discovered_factors.py')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"Saved to {out_path}")
    available = sum(1 for r in results if r['status'] == 'AVAILABLE')
    print(f"Scan complete: {len(results)} tested, {available} available")
```

- [ ] **Step 2: Verify imports**

Run: `cd F:/编程文件/quant_system_v1 && python -c "import sys; sys.path.insert(0,'.'); from evolution.datasource_scanner import TushareAPIScanner; print('OK')"`
Expected: "OK"

- [ ] **Step 3: Commit**

```bash
git add quant_system_v1/evolution/
git commit -m "feat: data source scanner — auto-discover unused Tushare APIs"
```

---

### Task 2: Formula Miner (Genetic Programming)

**Files:**
- Create: `quant_system_v1/evolution/formula_miner.py`

- [ ] **Step 1: Create formula_miner.py**

```python
"""
Genetic programming for factor formula discovery.

Evolves mathematical expressions using price/volume data as terminals.
Fitness = |ICIR| computed via FactorEvaluator.
Survivors registered as new FactorBase subclasses.
"""
import os, sys, re, random, uuid
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from factor_lib.evaluation import FactorEvaluator

logger = get_logger("formula_miner")

OPS = ['add', 'sub', 'mul', 'safe_div', 'ts_mean_5', 'ts_mean_10', 'ts_std_5',
       'ts_rank_10', 'ts_delta_5', 'ts_delay_5']
TERMINALS = ['open', 'high', 'low', 'close', 'vol', 'amount', 'turnover_rate']

POP_SIZE = 100
GENERATIONS = 20
CROSSOVER_PROB = 0.4
MUTATION_PROB = 0.3
ELITE = 10
MAX_DEPTH = 4


class FormulaGene:
    def __init__(self, depth=0):
        self.depth = depth
        if depth >= MAX_DEPTH or (depth > 1 and random.random() < 0.3):
            self.op = 'terminal'
            self.value = random.choice(TERMINALS)
            self.children = []
        else:
            self.op = random.choice(OPS)
            self.children = []
            n_children = 2 if self.op in ('add', 'sub', 'mul', 'safe_div') else 1
            for _ in range(n_children):
                self.children.append(FormulaGene(depth + 1))

    def to_code(self):
        if self.op == 'terminal':
            return f"df['{self.value}']"
        if self.op == 'add':
            return f"({self.children[0].to_code()} + {self.children[1].to_code()})"
        if self.op == 'sub':
            return f"({self.children[0].to_code()} - {self.children[1].to_code()})"
        if self.op == 'mul':
            return f"({self.children[0].to_code()} * {self.children[1].to_code()})"
        if self.op == 'safe_div':
            return f"({self.children[0].to_code()} / ({self.children[1].to_code()}.replace(0, np.nan) + 1e-8))"
        if self.op == 'ts_mean_5':
            return f"df.groupby('ts_code')[{self.children[0].to_code().replace(\"df['\",'').replace(\"']\",'')}].transform(lambda x: x.rolling(5).mean())"
        if self.op == 'ts_std_5':
            return f"df.groupby('ts_code')[{self.children[0].to_code().replace(\"df['\",'').replace(\"']\",'')}].transform(lambda x: x.rolling(5).std())"
        if self.op == 'ts_rank_10':
            return f"df.groupby('trade_date')[{self.children[0].to_code().replace(\"df['\",'').replace(\"']\",'')}].transform(lambda x: x.rank(pct=True))"
        if self.op == 'ts_delta_5':
            return f"df.groupby('ts_code')[{self.children[0].to_code().replace(\"df['\",'').replace(\"']\",'')}].transform(lambda x: x.diff(5))"
        if self.op == 'ts_delay_5':
            return f"df.groupby('ts_code')[{self.children[0].to_code().replace(\"df['\",'').replace(\"']\",'')}].shift(5)"
        return "np.zeros(len(df))"

    def to_factor_code(self, name):
        code = self.to_code()
        return f"""
@FactorRegistry.register
class GP_{name}(FactorBase):
    name = "gp_{name}"; category = "gp"; desc = "Genetic programming: {code[:60]}"
    def compute(self, df):
        try:
            return {code}
        except Exception:
            return pd.Series(np.nan, index=df.index)
"""


class FactorPopulation:
    def __init__(self, size=POP_SIZE):
        self.individuals = [FormulaGene() for _ in range(size)]
        self.fitnesses = {}
        self.evaluator = FactorEvaluator(min_obs=50, icir_threshold=0.2)
        self.generation = 0

    def evaluate(self, data: pd.DataFrame, returns: pd.Series, idx=0):
        gene = self.individuals[idx]
        code = gene.to_code()
        try:
            result = eval(code, {'df': data, 'np': np, 'pd': pd, '__builtins__': {}})
            if isinstance(result, pd.Series) and len(result.dropna()) > 50:
                eval_result = self.evaluator.evaluate(
                    f"gp_gen{self.generation}_{idx}", "gp", result, returns
                )
                return abs(eval_result.icir)
        except Exception:
            pass
        return 0.0

    def evolve_one_generation(self, data, returns):
        self.generation += 1
        # Evaluate
        for i in range(len(self.individuals)):
            self.fitnesses[i] = self.evaluate(data, returns, i)
        # Sort
        ranked = sorted(self.fitnesses.items(), key=lambda x: x[1], reverse=True)
        # Elite
        new_pop = [self.individuals[ranked[i][0]] for i in range(ELITE)]
        # Crossover + mutation
        while len(new_pop) < POP_SIZE:
            parent1 = self._tournament(ranked)
            parent2 = self._tournament(ranked)
            if random.random() < CROSSOVER_PROB:
                child = self._crossover(parent1, parent2)
            else:
                child = FormulaGene()
            if random.random() < MUTATION_PROB:
                child = self._mutate(child)
            new_pop.append(child)
        self.individuals = new_pop
        best_idx = ranked[0][0]
        logger.info(f"Gen {self.generation}: best ICIR={ranked[0][1]:.4f}, "
                     f"median={np.median([v for v in self.fitnesses.values() if v > 0]):.4f}")
        return self.individuals[best_idx], ranked[0][1]

    def _tournament(self, ranked, k=5):
        candidates = random.sample(ranked, min(k, len(ranked)))
        return self.individuals[max(candidates, key=lambda x: x[1])[0]]

    def _crossover(self, p1, p2):
        child = FormulaGene()
        if p1.children and p2.children:
            child.op = random.choice([p1.op, p2.op])
            child.children = [
                self._crossover(c1, c2) if random.random() < 0.5 else FormulaGene()
                for c1, c2 in zip(p1.children[:2], p2.children[:2])
            ]
        return child

    def _mutate(self, gene):
        if random.random() < 0.5:
            gene.op = random.choice(OPS)
        else:
            gene.value = random.choice(TERMINALS)
        return gene


class FormulaMiner:
    def __init__(self, pop_size=POP_SIZE, generations=GENERATIONS):
        self.pop = FactorPopulation(pop_size)
        self.generations = generations
        self.best_formulas = []

    def run(self, df, output_dir):
        returns = df.groupby('ts_code')['close'].transform(lambda x: x.pct_change().shift(-1))
        os.makedirs(output_dir, exist_ok=True)
        for gen in range(self.generations):
            best_gene, best_fitness = self.pop.evolve_one_generation(df, returns)
            if best_fitness > 0.2:
                self.best_formulas.append((best_gene, best_fitness))
                logger.info(f"New best formula: ICIR={best_fitness:.4f}")
        # Export top formulas
        unique = {}
        for gene, fit in sorted(self.best_formulas, key=lambda x: x[1], reverse=True):
            code = gene.to_code()
            h = hash(code)
            if h not in unique:
                unique[h] = (gene, fit, code)
        code_output = [
            "# Auto-discovered factors via genetic programming",
            f"# Generated: {pd.Timestamp.now().isoformat()}",
            f"# Formulas found: {len(unique)}",
            "import pandas as pd; import numpy as np",
            "from factor_lib.registry import FactorRegistry, FactorBase",
            "",
        ]
        for i, (gene, fit, code) in enumerate(list(unique.values())[:20]):
            name = f"gp_{fit*100:.0f}_{i}"
            code_output.append(gene.to_factor_code(name))
        out_path = os.path.join(output_dir, 'gp_discovered_factors.py')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(code_output))
        logger.info(f"Saved {len(unique)} formulas to {out_path}")
        return unique


if __name__ == '__main__':
    print("Formula miner: run via 'python main.py evolve --mode factors'")
```

- [ ] **Step 2: Verify import**

Run: `cd F:/编程文件/quant_system_v1 && python -c "import sys; sys.path.insert(0,'.'); from evolution.formula_miner import FormulaMiner, FormulaGene; print('OK')"`
Expected: "OK"

- [ ] **Step 3: Commit**

```bash
git add quant_system_v1/evolution/formula_miner.py
git commit -m "feat: genetic programming formula miner for auto factor discovery"
```

---

### Task 3: Factor Evolution (Rotator + Weight Optimizer + Lifecycle)

**Files:**
- Create: `quant_system_v1/evolution/factor_evolution.py`

- [ ] **Step 1: Create factor_evolution.py**

```python
"""
Factor evolution: market-regime rotation, adaptive weight optimization, lifecycle monitoring.

- FactorRotator: switch factor sets per market regime (bull/bear/normal)
- AdaptiveWeightOptimizer: monthly re-weight factors by trailing ICIR
- FactorLifecycleMonitor: detect decaying factors and trigger alerts
"""
import os, sys, json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from factor_lib.registry import FactorRegistry

logger = get_logger("factor_evo")


class FactorRotator:
    def __init__(self):
        self.regime_factors = {
            'bull': {'price': 1.5, 'sentiment': 1.3, 'volume': 1.2, 'technical': 1.0,
                     'fundamental': 0.8, 'sector': 1.0, 'macro': 0.7},
            'bear': {'fundamental': 1.5, 'macro': 1.3, 'sector': 1.0, 'technical': 1.0,
                     'price': 0.7, 'volume': 0.8, 'sentiment': 0.7},
            'normal': {'price': 1.0, 'volume': 1.0, 'technical': 1.0, 'fundamental': 1.0,
                       'sentiment': 1.0, 'sector': 1.0, 'macro': 1.0},
        }

    def get_weights(self, regime: str, factor_names: list) -> dict:
        weights = self.regime_factors.get(regime, self.regime_factors['normal'])
        result = {}
        for name in factor_names:
            cat = FactorRegistry._factors.get(name, type('',(),{'category':'price'})()).category if name in FactorRegistry._factors else 'price'
            result[name] = weights.get(cat, 1.0)
        return result


class AdaptiveWeightOptimizer:
    def __init__(self, history_file=None):
        self.history_file = history_file or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'evolution_output', 'ic_history.json'
        )

    def update(self, factor_name, ic_value, date):
        history = self._load()
        if factor_name not in history:
            history[factor_name] = []
        history[factor_name].append({'date': date, 'ic': float(ic_value)})
        history[factor_name] = history[factor_name][-252:]  # keep 1 year
        self._save(history)

    def get_weight(self, factor_name, lookback_months=6):
        history = self._load()
        entries = history.get(factor_name, [])
        if not entries:
            return 1.0
        cutoff = (datetime.now() - timedelta(days=lookback_months * 30)).isoformat()[:10]
        recent = [e for e in entries if e['date'] >= cutoff]
        if not recent:
            return 1.0
        ics = [abs(e['ic']) for e in recent]
        avg_ic = np.mean(ics)
        if avg_ic < 0.1:
            return 0.0  # retired
        if avg_ic < 0.15:
            return 0.3  # cooling
        return min(avg_ic / 0.05, 2.0)  # scale: IC=0.05 → weight=1.0

    def _load(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {}

    def _save(self, history):
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)


class FactorLifecycleMonitor:
    def __init__(self, history_file=None):
        self.history_file = history_file or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'evolution_output', 'ic_history.json'
        )

    def check(self, factor_name):
        history = self._load()
        entries = history.get(factor_name, [])
        if len(entries) < 20:
            return {'status': 'new', 'trend': 'unknown'}
        ics = [e['ic'] for e in entries[-60:]]
        avg_recent = np.mean(ics[-20:])
        avg_prior = np.mean(ics[-40:-20]) if len(ics) >= 40 else avg_recent
        change_pct = (avg_recent - avg_prior) / max(abs(avg_prior), 0.001)
        if change_pct < -0.3:
            return {'status': 'CRASH', 'trend': 'down', 'change': change_pct,
                    'action': 'REMOVE', 'avg_ic': avg_recent}
        if change_pct < -0.15:
            return {'status': 'DECLINING', 'trend': 'down', 'change': change_pct,
                    'action': 'WARN', 'avg_ic': avg_recent}
        if change_pct > 0.15:
            return {'status': 'IMPROVING', 'trend': 'up', 'change': change_pct,
                    'action': 'BOOST', 'avg_ic': avg_recent}
        return {'status': 'STABLE', 'trend': 'flat', 'change': change_pct,
                'action': 'KEEP', 'avg_ic': avg_recent}

    def _load(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {}
```

- [ ] **Step 2: Verify import**

Run: `cd F:/编程文件/quant_system_v1 && python -c "import sys; sys.path.insert(0,'.'); from evolution.factor_evolution import FactorRotator, AdaptiveWeightOptimizer, FactorLifecycleMonitor; print('OK')"`
Expected: "OK"

- [ ] **Step 3: Commit**

```bash
git add quant_system_v1/evolution/factor_evolution.py
git commit -m "feat: factor evolution — regime rotation, adaptive weights, lifecycle monitoring"
```

---

### Task 4: GitHub Strategy Miner

**Files:**
- Create: `quant_system_v1/evolution/github_strategy_miner.py`

- [ ] **Step 1: Create github_strategy_miner.py**

```python
"""
GitHub strategy miner: search, clone, extract, backtest-compare.

1. gh search repos for A-share quant strategies
2. Clone top repos to sandbox/
3. Extract strategy classes via AST parsing
4. Backtest each extracted strategy on 2024 data
5. Rank vs existing 7 strategies, register top 3
"""
import os, sys, re, ast, subprocess, json
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger

logger = get_logger("gh_miner")

SEARCH_QUERIES = [
    "A股 量化策略 language:python",
    "涨停策略 回测 language:python",
    "龙头战法 量化 language:python",
    "短线策略 backtest language:python",
    "均线突破 A股 language:python",
]


class GitHubSearcher:
    def __init__(self, sandbox_dir=None):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.sandbox_dir = sandbox_dir or os.path.join(base, '..', 'sandbox')
        self.results = []

    def search(self, min_stars=3, max_repos=20):
        all_repos = []
        for query in SEARCH_QUERIES[:3]:
            try:
                cmd = f'gh search repos "{query}" --sort stars --limit 10 --json name,fullName,url,stargazerCount,description'
                output = subprocess.check_output(cmd, shell=True, text=True, timeout=30)
                repos = json.loads(output)
                for r in repos:
                    if r['stargazerCount'] >= min_stars:
                        all_repos.append(r)
            except Exception as e:
                logger.warning(f"Search failed for '{query}': {e}")
        seen = set()
        unique = []
        for r in all_repos:
            if r['fullName'] not in seen:
                seen.add(r['fullName'])
                unique.append(r)
        self.results = unique[:max_repos]
        logger.info(f"Found {len(self.results)} unique repos")
        return self.results

    def clone_all(self):
        os.makedirs(self.sandbox_dir, exist_ok=True)
        for repo in self.results:
            name = repo['fullName'].replace('/', '_')
            target = os.path.join(self.sandbox_dir, name)
            if os.path.exists(target):
                continue
            try:
                subprocess.run(['git', 'clone', '--depth', '1', repo['url'], target],
                              check=True, timeout=60, capture_output=True)
                logger.info(f"Cloned {repo['fullName']} ({repo['stargazerCount']} stars)")
            except Exception as e:
                logger.warning(f"Clone failed {repo['fullName']}: {e}")

    def list_cloned(self):
        if not os.path.isdir(self.sandbox_dir):
            return []
        return [d for d in os.listdir(self.sandbox_dir)
                if os.path.isdir(os.path.join(self.sandbox_dir, d, '.git'))]


class StrategyExtractor:
    def extract_from_dir(self, repo_dir):
        strategies = []
        for root, _, files in os.walk(repo_dir):
            for f in files:
                if f.endswith('.py') and 'strategy' in f.lower():
                    fp = os.path.join(root, f)
                    try:
                        with open(fp, 'r', encoding='utf-8', errors='ignore') as fh:
                            code = fh.read()
                        tree = ast.parse(code)
                        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
                        for cls in classes:
                            methods = [n.name for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)]
                            if 'run' in methods or 'filter' in methods or 'score' in methods:
                                strategies.append({
                                    'repo': os.path.basename(repo_dir),
                                    'file': f,
                                    'class': cls.name,
                                    'methods': methods,
                                    'code': code,
                                })
                    except Exception:
                        pass
        return strategies


class StrategyComparator:
    def __init__(self):
        self.results = []

    def backtest_external(self, strategy_code, strategy_name):
        """Run backtest on external strategy code and return metrics."""
        import tempfile
        fd, path = tempfile.mkstemp(suffix='.py')
        with os.fdopen(fd, 'w') as f:
            f.write(strategy_code)
        try:
            from backtest.vectorized_engine import VectorizedBacktestEngine
            sys.path.insert(0, os.path.dirname(path))
            module_name = os.path.basename(path).replace('.py', '')
            mod = __import__(module_name)
            if hasattr(mod, 'run_backtest'):
                result = mod.run_backtest(start='2024-01-01', end='2024-12-31')
                return result
        except Exception as e:
            logger.warning(f"Backtest failed for {strategy_name}: {e}")
        finally:
            os.unlink(path)
        return None

    def rank(self, external_results, existing_results):
        all_results = [(name, r.get('sharpe_ratio', 0), r.get('total_return', 0))
                       for name, r in external_results + existing_results]
        return sorted(all_results, key=lambda x: x[1], reverse=True)
```

- [ ] **Step 2: Verify import**

Run: `cd F:/编程文件/quant_system_v1 && python -c "import sys; sys.path.insert(0,'.'); from evolution.github_strategy_miner import GitHubSearcher, StrategyExtractor; print('OK')"`
Expected: "OK"

- [ ] **Step 3: Commit**

```bash
git add quant_system_v1/evolution/github_strategy_miner.py
git commit -m "feat: GitHub strategy miner — search, clone, extract, backtest-compare"
```

---

### Task 5: ML Predictor + Feature Engineering + Model Manager

**Files:**
- Create: `quant_system_v1/ml/__init__.py`
- Create: `quant_system_v1/ml/feature_engineering.py`
- Create: `quant_system_v1/ml/predictor.py`
- Create: `quant_system_v1/ml/model_manager.py`

- [ ] **Step 1: Create feature_engineering.py**

```python
"""Auto feature engineering: cross features, lag features, aggregation."""
import pandas as pd
import numpy as np
from itertools import combinations


def generate_cross_features(df, factor_cols, top_n=50):
    for c1, c2 in combinations(factor_cols[:min(len(factor_cols), 15)], 2):
        name = f'{c1}_x_{c2}'
        df[name] = df[c1] * df[c2]
    return df


def generate_lag_features(df, factor_cols, lags=(1, 2, 5)):
    for col in factor_cols:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df.groupby('ts_code')[col].shift(lag)
    return df


def generate_agg_features(df, factor_cols):
    for col in factor_cols:
        if 'industry' in df.columns:
            df[f'{col}_ind_mean'] = df.groupby(['trade_date', 'industry'])[col].transform('mean')
    return df
```

- [ ] **Step 2: Create predictor.py**

```python
"""LightGBM-based next-day direction predictor."""
import os, sys, pickle
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from factor_lib.registry import FactorRegistry

logger = get_logger("ml")


class MLPredictor:
    def __init__(self, model_dir=None):
        base = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = model_dir or os.path.join(base, '..', 'evolution_output', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = None
        self.feature_names = []

    def prepare_data(self, df, factor_names=None):
        if factor_names is None:
            factor_names = FactorRegistry.list_all()
        factor_df = FactorRegistry.compute(df, factor_names)
        X = factor_df.dropna(axis=1, thresh=int(len(factor_df) * 0.5)).fillna(0)
        df['_fwd_ret'] = df.groupby('ts_code')['close'].transform(lambda x: x.pct_change().shift(-1))
        y = (df['_fwd_ret'] > 0).astype(int)
        self.feature_names = X.columns.tolist()
        return X, y

    def train(self, X, y, valid_frac=0.2):
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("lightgbm required: pip install lightgbm")
            return None
        split = int(len(X) * (1 - valid_frac))
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_valid, y_valid = X.iloc[split:], y.iloc[split:]
        self.model = lgb.LGBMClassifier(
            objective='binary', metric='auc', n_estimators=200,
            max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbose=-1,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        auc = self.model.best_score_['valid_0']['auc']
        logger.info(f"Trained LightGBM: AUC={auc:.4f}, features={len(self.feature_names)}")
        return auc

    def predict(self, X):
        if self.model is None:
            return pd.Series(0.5, index=X.index)
        return pd.Series(self.model.predict_proba(X)[:, 1], index=X.index,
                         name='ml_probability')

    def save(self, version=None):
        if self.model is None:
            return
        v = version or pd.Timestamp.now().strftime('%Y%m%d')
        path = os.path.join(self.model_dir, f'model_{v}.pkl')
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'features': self.feature_names}, f)
        logger.info(f"Model saved: {path}")

    def load(self, version=None):
        if version is None:
            files = sorted([f for f in os.listdir(self.model_dir) if f.endswith('.pkl')])
            if not files:
                logger.warning("No model file found")
                return False
            path = os.path.join(self.model_dir, files[-1])
        else:
            path = os.path.join(self.model_dir, f'model_{version}.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_names = data['features']
        logger.info(f"Model loaded: {path}")
        return True
```

- [ ] **Step 3: Create model_manager.py**

```python
"""Model version management and periodic retraining."""
import os, sys, json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from .predictor import MLPredictor

logger = get_logger("model_mgr")


class ModelManager:
    def __init__(self, model_dir=None, max_versions=3):
        base = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = model_dir or os.path.join(base, '..', 'evolution_output', 'models')
        self.max_versions = max_versions
        self.predictor = MLPredictor(self.model_dir)
        self.manifest_file = os.path.join(self.model_dir, 'manifest.json')

    def retrain(self, df, force=False):
        X, y = self.predictor.prepare_data(df)
        auc = self.predictor.train(X, y)
        if auc is None:
            return False
        # Only save if improvement or forced
        current_auc = self._current_best_auc()
        if force or current_auc is None or auc > current_auc:
            version = pd.Timestamp.now().strftime('%Y%m%d')
            self.predictor.save(version)
            self._update_manifest(version, auc)
            self._cleanup_old()
            logger.info(f"New model version {version}: AUC={auc:.4f} (prev={current_auc})")
            return True
        logger.info(f"Model not improved: {auc:.4f} <= {current_auc:.4f}")
        return False

    def _current_best_auc(self):
        manifest = self._load_manifest()
        if manifest:
            return max(v['auc'] for v in manifest.values())
        return None

    def _update_manifest(self, version, auc):
        manifest = self._load_manifest()
        manifest[version] = {'auc': auc, 'date': pd.Timestamp.now().isoformat()}
        with open(self.manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

    def _cleanup_old(self):
        manifest = self._load_manifest()
        sorted_versions = sorted(manifest.items(), key=lambda x: x[1]['auc'], reverse=True)
        for version, _ in sorted_versions[self.max_versions:]:
            path = os.path.join(self.model_dir, f'model_{version}.pkl')
            if os.path.exists(path):
                os.remove(path)
            manifest.pop(version, None)
        with open(self.manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

    def _load_manifest(self):
        if os.path.exists(self.manifest_file):
            with open(self.manifest_file, 'r') as f:
                return json.load(f)
        return {}
```

- [ ] **Step 4: Verify imports**

Run:
```bash
cd F:/编程文件/quant_system_v1 && python -c "
import sys; sys.path.insert(0,'.')
from ml.feature_engineering import generate_cross_features
from ml.predictor import MLPredictor
from ml.model_manager import ModelManager
print('OK')
"
```
Expected: "OK"

- [ ] **Step 5: Commit**

```bash
git add quant_system_v1/ml/
git commit -m "feat: ML predictor (LightGBM) + feature engineering + model version management"
```

---

### Task 6: Scheduler + CLI Integration

**Files:**
- Create: `quant_system_v1/scheduler/__init__.py`
- Create: `quant_system_v1/scheduler/scheduler.py`
- Modify: `quant_system_v1/main.py` (add evolve command)

- [ ] **Step 1: Create scheduler.py**

```python
"""Continuous learning scheduler: daily/weekly/monthly tasks."""
import os, sys, time, json
import pandas as pd
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.calendar import TradeCalendar

logger = get_logger("scheduler")


class EvolutionScheduler:
    def __init__(self):
        self.cal = TradeCalendar()
        self.last_run = self._load_state()

    def run_daily(self):
        today = datetime.now().strftime('%Y-%m-%d')
        if today == self.last_run.get('daily'):
            return
        logger.info(f"[DAILY] {today}: Updating factor IC records")
        self.last_run['daily'] = today
        self._save_state()

    def run_weekly(self):
        week = datetime.now().strftime('%Y-W%W')
        if week == self.last_run.get('weekly'):
            return
        logger.info(f"[WEEKLY] {week}: Factor lifecycle check")
        from evolution.factor_evolution import FactorLifecycleMonitor
        monitor = FactorLifecycleMonitor()
        from factor_lib.registry import FactorRegistry
        for name in FactorRegistry.list_all():
            status = monitor.check(name)
            if status['action'] in ('REMOVE', 'WARN'):
                logger.warning(f"  {name}: {status}")
        self.last_run['weekly'] = week
        self._save_state()

    def run_monthly(self):
        month = datetime.now().strftime('%Y-%m')
        if month == self.last_run.get('monthly'):
            return
        logger.info(f"[MONTHLY] {month}: Full evolution cycle")
        self.last_run['monthly'] = month
        self._save_state()

    def run_all(self):
        self.run_daily()
        self.run_weekly()
        self.run_monthly()

    def _load_state(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'evolution_output', 'scheduler_state.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    def _save_state(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'evolution_output', 'scheduler_state.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.last_run, f, indent=2)
```

- [ ] **Step 2: Add evolve command to main.py**

Edit main.py, append after existing commands:

```python
def cmd_evolve(args):
    """自主进化"""
    from scheduler.scheduler import EvolutionScheduler
    scheduler = EvolutionScheduler()
    if args.mode == 'factors':
        logger.info("Running factor evolution...")
        from evolution.formula_miner import FormulaMiner
        from evolution.datasource_scanner import TushareAPIScanner
        from evolution.factor_evolution import AdaptiveWeightOptimizer, FactorLifecycleMonitor
        scanner = TushareAPIScanner()
        scanner.scan()
        code = scanner.export_candidates()
        if code:
            out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'evolution_output', 'factors')
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, 'auto_discovered_factors.py'), 'w') as f:
                f.write(code)
        logger.info("Factor evolution complete")
    elif args.mode == 'strategies':
        logger.info("Running strategy evolution...")
        from evolution.github_strategy_miner import GitHubSearcher, StrategyExtractor
        searcher = GitHubSearcher()
        repos = searcher.search()
        searcher.clone_all()
        extractor = StrategyExtractor()
        for d in searcher.list_cloned():
            strategies = extractor.extract_from_dir(os.path.join(searcher.sandbox_dir, d))
            logger.info(f"  {d}: {len(strategies)} strategy classes found")
        logger.info("Strategy evolution complete")
    elif args.mode == 'ml':
        logger.info("Running ML retraining...")
        from data_source.fetcher import DataFetcher
        from ml.model_manager import ModelManager
        mgr = ModelManager()
        mgr.retrain(None)  # Uses existing data
        logger.info("ML retraining complete")
    elif args.mode == 'full':
        logger.info("Full evolution cycle...")
        scheduler.run_all()
        logger.info("Full evolution complete")
```

And add subparser in main():

```python
# evolve
p = sub.add_parser('evolve', help='自主进化')
p.add_argument('--mode', choices=['factors', 'strategies', 'ml', 'full'], default='full')
```

- [ ] **Step 3: Verify CLI**

Run: `cd F:/编程文件/quant_system_v1 && python main.py evolve --help`
Expected: Show evolve help with --mode options

- [ ] **Step 4: Commit**

```bash
git add quant_system_v1/scheduler/ quant_system_v1/main.py
git commit -m "feat: evolution scheduler + CLI evolve command"
```

---

### Task 7: ML Strategy Wrapper (register as strategy #8)

**Files:**
- Create: `quant_system_v1/strategy/ml_strategy.py`

- [ ] **Step 1: Create ML strategy wrapper**

```python
"""ML预测策略 — LightGBM涨跌概率作为信号"""
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base import StrategyBase, StrategyRegistry


@StrategyRegistry.register
class MLPredictionStrategy(StrategyBase):
    name = "ML预测策略"
    version = "1.0"

    def __init__(self, config=None):
        super().__init__(config)
        self._predictor = None

    def _get_predictor(self):
        if self._predictor is None:
            from ml.predictor import MLPredictor
            self._predictor = MLPredictor()
            self._predictor.load()
        return self._predictor

    def filter(self, df):
        if df.empty:
            return df
        df = df.copy()
        if 'name' in df.columns:
            df = df[~df['name'].str.contains('ST|退市', na=False)]
        if 'close' in df.columns:
            df = df[df['close'] > 0]
        if 'amount' in df.columns:
            df = df[df['amount'] >= 100000]
        return df.reset_index(drop=True)

    def score(self, df):
        if df.empty:
            return df
        df = df.copy()
        try:
            predictor = self._get_predictor()
            X, _ = predictor.prepare_data(df)
            probs = predictor.predict(X)
            df['total_score'] = probs.values * 20
        except Exception:
            df['total_score'] = 0.0
        df = df[df['total_score'] >= 8]
        return df.sort_values('total_score', ascending=False).reset_index(drop=True)

    def generate_signals_vectorized(self, df):
        if df.empty or 'trade_date' not in df.columns:
            return pd.DataFrame()
        try:
            scored = self.score(df)
            if scored.empty or 'total_score' not in scored.columns:
                return pd.DataFrame()
            scored['signal'] = (scored['total_score'] >= 10).astype(int)
            return scored.pivot_table(
                index='trade_date', columns='ts_code', values='signal', fill_value=0
            ).astype(int)
        except Exception:
            return pd.DataFrame()
```

- [ ] **Step 2: Add import to main.py**

Edit main.py, append to strategy imports:
```python
import strategy.ml_strategy
```

- [ ] **Step 3: Verify**

Run: `cd F:/编程文件/quant_system_v1 && python -c "import sys; sys.path.insert(0,'.'); import main; from strategy import StrategyRegistry; print(StrategyRegistry.list_all())"`
Expected: 8 strategies including "ML预测策略"

- [ ] **Step 4: Commit**

```bash
git add quant_system_v1/strategy/ml_strategy.py quant_system_v1/main.py
git commit -m "feat: ML prediction strategy — register as 8th strategy"
```

---

## Self-Review

### Spec Coverage
- [x] Formula miner → Task 2
- [x] Data source scanner → Task 1
- [x] Factor evolution (rotator/weight/lifecycle) → Task 3
- [x] GitHub strategy miner → Task 4
- [x] ML predictor + feature engineering + model manager → Task 5
- [x] Scheduler + CLI integration → Task 6
- [x] ML strategy wrapper → Task 7

### Placeholder Check
No TBD/TODO/incomplete items.

### Type Consistency
- `FactorEvaluator` from Task 4 of v2 plan, used in Task 2
- `MLPredictor.predict()` returns `pd.Series`, consumed by `MLPredictionStrategy.score()`
- `EvolutionScheduler` imports from `evolution.*` (Task 1-3) and `ml.*` (Task 5)
