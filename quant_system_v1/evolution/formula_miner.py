"""
Genetic programming for factor formula discovery.

Evolves mathematical expressions using price/volume data as terminals.
Fitness = |ICIR| computed via FactorEvaluator.
Survivors exported as auto-generated FactorBase subclasses.
"""
import os, sys, random
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from factor_lib.evaluation import FactorEvaluator

logger = get_logger("formula_miner")

OPS = ['add', 'sub', 'mul', 'safe_div', 'ts_mean_5', 'ts_std_5', 'ts_rank_10', 'ts_delta_5']
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
            n = 2 if self.op in ('add', 'sub', 'mul', 'safe_div') else 1
            for _ in range(n):
                self.children.append(FormulaGene(depth + 1))

    def to_code(self):
        if self.op == 'terminal':
            col = self.value
            return f"df['{col}']"
        c = self.children
        if self.op == 'add':
            return f"({c[0].to_code()} + {c[1].to_code()})"
        if self.op == 'sub':
            return f"({c[0].to_code()} - {c[1].to_code()})"
        if self.op == 'mul':
            return f"({c[0].to_code()} * {c[1].to_code()})"
        if self.op == 'safe_div':
            return f"({c[0].to_code()} / ({c[1].to_code()}.replace(0, np.nan) + 1e-8))"
        col = c[0].to_code()
        col_name = col.replace("df['", "").replace("']", "")
        if self.op == 'ts_mean_5':
            return (f"df.groupby('ts_code')[{col}].transform(lambda x: x.rolling(5, min_periods=1).mean())")
        if self.op == 'ts_std_5':
            return (f"df.groupby('ts_code')[{col}].transform(lambda x: x.rolling(5, min_periods=1).std())")
        if self.op == 'ts_rank_10':
            return (f"df.groupby('trade_date')[{col}].transform(lambda x: x.rank(pct=True))")
        if self.op == 'ts_delta_5':
            return (f"df.groupby('ts_code')[{col}].transform(lambda x: x.diff(5))")
        return "np.zeros(len(df))"

    def to_factor_code(self, name):
        code = self.to_code()
        snippet = code[:60]
        return f"""
@FactorRegistry.register
class GP_{name}(FactorBase):
    name = "gp_{name}"; category = "gp"; desc = "GP: {snippet}"
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

    def evaluate_one(self, data, returns, idx):
        gene = self.individuals[idx]
        code = gene.to_code()
        try:
            result = eval(code, {'df': data, 'np': np, 'pd': pd, '__builtins__': {}})
            if isinstance(result, pd.Series) and len(result.dropna()) > 50:
                ev = self.evaluator.evaluate(f"gp_gen{self.generation}_{idx}", "gp", result, returns)
                return abs(ev.icir)
        except Exception:
            pass
        return 0.0

    def evolve_one_generation(self, data, returns):
        self.generation += 1
        for i in range(len(self.individuals)):
            self.fitnesses[i] = self.evaluate_one(data, returns, i)
        ranked = sorted(self.fitnesses.items(), key=lambda x: x[1], reverse=True)
        new_pop = [self.individuals[ranked[i][0]] for i in range(min(ELITE, len(ranked)))]
        while len(new_pop) < POP_SIZE:
            p1 = self._tournament(ranked)
            p2 = self._tournament(ranked)
            child = self._crossover(p1, p2) if random.random() < CROSSOVER_PROB else FormulaGene()
            if random.random() < MUTATION_PROB:
                child = self._mutate(child)
            new_pop.append(child)
        self.individuals = new_pop
        best_idx = ranked[0][0]
        positive = [v for v in self.fitnesses.values() if v > 0]
        logger.info(f"Gen {self.generation}: best ICIR={ranked[0][1]:.4f}, "
                     f"median={np.median(positive) if positive else 0:.4f}")
        return self.individuals[best_idx], ranked[0][1]

    def _tournament(self, ranked, k=5):
        candidates = random.sample(ranked, min(k, len(ranked)))
        return self.individuals[max(candidates, key=lambda x: x[1])[0]]

    def _crossover(self, p1, p2):
        child = FormulaGene()
        if p1.children and p2.children:
            child.op = random.choice([p1.op, p2.op])
            child.children = []
            for c1, c2 in zip(p1.children[:2], p2.children[:2]):
                child.children.append(self._crossover(c1, c2) if random.random() < 0.5 else FormulaGene())
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
            code_output.append(gene.to_factor_code(f"{fit*100:.0f}_{i}"))
        out_path = os.path.join(output_dir, 'gp_discovered_factors.py')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(code_output))
        logger.info(f"Saved {len(unique)} formulas to {out_path}")
        return unique
