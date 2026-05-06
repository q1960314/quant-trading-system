"""
DuckDB-based data store for quant data.

Replaces CSV directory scanning with SQL queries.
Single-file database, zero-config, columnar storage.
"""
import os
import pandas as pd
from utils.logger import get_logger

logger = get_logger("duckdb_store")

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


class DuckDBStore:
    def __init__(self, db_path: str = None):
        if not HAS_DUCKDB:
            raise ImportError("DuckDB required: pip install duckdb")
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                '..', 'data', 'quant.duckdb',
            )
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = duckdb.connect(self.db_path)

    def create_table_from_csv(self, table_name: str, csv_path: str):
        if not os.path.exists(csv_path):
            logger.warning(f"CSV not found: {csv_path}")
            return
        self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        safe_path = csv_path.replace('\\', '/')
        self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{safe_path}')")
        count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        logger.info(f"Imported {count} rows into {table_name}")

    def query(self, sql: str) -> pd.DataFrame:
        return self.conn.execute(sql).df()

    def get_daily(self, codes=None, start=None, end=None) -> pd.DataFrame:
        conditions = []
        if codes:
            code_list = "','".join(str(c) for c in codes)
            conditions.append(f"ts_code IN ('{code_list}')")
        if start:
            conditions.append(f"trade_date >= '{start}'")
        if end:
            conditions.append(f"trade_date <= '{end}'")
        where = " AND ".join(conditions) if conditions else "1=1"
        return self.query(f"SELECT * FROM daily WHERE {where} ORDER BY trade_date, ts_code")

    def import_all_csvs(self, data_dir: str):
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        for csv_file in csv_files:
            table_name = csv_file.replace('.csv', '')
            csv_path = os.path.join(data_dir, csv_file)
            logger.info(f"Importing {table_name} from {csv_file}...")
            self.create_table_from_csv(table_name, csv_path)
        logger.info(f"Imported {len(csv_files)} tables")

    def close(self):
        self.conn.close()
