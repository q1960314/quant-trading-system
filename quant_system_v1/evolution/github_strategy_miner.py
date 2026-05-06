"""
GitHub strategy miner: search, clone, extract, backtest-compare.

1. gh search repos for A-share quant strategies
2. Clone top repos to sandbox/
3. Extract strategy classes via AST parsing
4. Backtest each on 2024 data, rank vs existing strategies
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
                logger.warning(f"Search failed: {e}")
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
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef):
                                methods = [n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
                                if any(m in methods for m in ('run', 'filter', 'score')):
                                    strategies.append({
                                        'repo': os.path.basename(repo_dir),
                                        'file': f, 'class': node.name,
                                        'methods': methods, 'code': code,
                                    })
                    except Exception:
                        pass
        return strategies


class StrategyComparator:
    def rank(self, external, existing):
        all_r = [(n, r.get('sharpe_ratio', 0), r.get('total_return', 0))
                 for n, r in external + existing]
        return sorted(all_r, key=lambda x: x[1], reverse=True)
