"""
Auto-scan Tushare API reference for unused data interfaces.

Parses tushare_api_reference.md, compares with existing CSV files,
tests each unused interface, and auto-registers new numeric columns as factors.
"""
import os, re, sys
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
        if not os.path.exists(API_REF_PATH):
            logger.warning(f"API doc not found: {API_REF_PATH}")
            return []
        with open(API_REF_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        matches = re.findall(r'###\s+(\w+)', content)
        return [m for m in matches if m not in ('接口', '概述', '目录')]

    def test_interface(self, api_name):
        try:
            df = self.mgr.call_api(api_name, start_date=SCAN_START, end_date=SCAN_END)
            if df is not None and not df.empty:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                return {
                    'api': api_name, 'status': 'AVAILABLE',
                    'rows': len(df),
                    'new_columns': [c for c in numeric_cols if c not in ('ts_code', 'trade_date')],
                }
            return {'api': api_name, 'status': 'EMPTY_DATA', 'rows': 0, 'new_columns': []}
        except Exception as e:
            return {'api': api_name, 'status': 'FAILED', 'error': str(e), 'new_columns': []}

    def scan(self, max_apis=50):
        api_list = self.parse_api_doc()
        if not api_list:
            logger.warning("No APIs parsed from doc")
            return self.results
        unused = [a for a in api_list if a not in self.existing_tables]
        logger.info(f"APIs: {len(api_list)} total, {len(unused)} unused, {len(self.existing_tables)} existing")
        for api_name in unused[:max_apis]:
            result = self.test_interface(api_name)
            self.results.append(result)
            if result['status'] == 'AVAILABLE' and result['new_columns']:
                logger.info(f"  {api_name}: {len(result['new_columns'])} new cols: {result['new_columns'][:5]}")
            else:
                logger.debug(f"  {api_name}: {result['status']}")
        return self.results

    def export_candidates(self):
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
    scanner.scan()
    code = scanner.export_candidates()
    if code:
        out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'evolution_output', 'factors', 'auto_discovered_factors.py')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"Saved to {out_path}")
    available = sum(1 for r in scanner.results if r['status'] == 'AVAILABLE')
    print(f"Scan complete: {len(scanner.results)} tested, {available} available")
