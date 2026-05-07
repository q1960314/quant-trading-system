"""Scan Tushare API doc for unused interfaces and test availability."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')

from evolution.datasource_scanner import TushareAPIScanner

scanner = TushareAPIScanner()
print(f"Existing tables: {len(scanner.existing_tables)}")
apis = scanner.parse_api_doc()
print(f"APIs in doc: {len(apis)}")
unused = [a for a in apis if a not in scanner.existing_tables]
print(f"Unused: {len(unused)}")

# Scan first 20 unused APIs (fast test)
print("\nTesting first 20 unused APIs...")
scanner.scan(max_apis=20)

code = scanner.export_candidates()
if code:
    out = os.path.join(os.path.dirname(__file__), '..', 'evolution_output', 'factors', 'auto_discovered_factors.py')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f"\nSaved discovered factors to {out}")

available = [r for r in scanner.results if r['status'] == 'AVAILABLE']
failed = [r for r in scanner.results if r['status'] == 'FAILED']
empty = [r for r in scanner.results if r['status'] == 'EMPTY_DATA']

print(f"\nResults: {len(available)} available, {len(empty)} empty, {len(failed)} failed")
for r in available:
    print(f"  + {r['api']}: {r['rows']} rows, {len(r['new_columns'])} new cols")
for r in failed:
    print(f"  - {r['api']}: {r.get('error','')[:60]}")
