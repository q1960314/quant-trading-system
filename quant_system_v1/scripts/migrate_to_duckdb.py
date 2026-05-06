"""
One-time script: migrate all CSV data to DuckDB.

Usage:
  python scripts/migrate_to_duckdb.py
  python scripts/migrate_to_duckdb.py --dry-run   # check without importing
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import duckdb
except ImportError:
    print("DuckDB required. Run: pip install duckdb")
    sys.exit(1)

from config.settings import LOCAL_GLOBAL_DIR, LOCAL_DATA_DIR


def migrate_global_csvs(db_path: str, data_dir: str, dry_run: bool = False):
    """Import all global CSVs (data/*.csv) into DuckDB."""
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    print(f"Found {len(csv_files)} global CSV files in {data_dir}")

    if dry_run:
        print("[DRY RUN] Would import:")
        for f in csv_files:
            fsize = os.path.getsize(os.path.join(data_dir, f))
            print(f"  {f}: {fsize/1024/1024:.1f} MB")
        return

    conn = duckdb.connect(db_path)
    for csv_file in csv_files:
        table_name = csv_file.replace('.csv', '')
        csv_path = os.path.join(data_dir, csv_file)
        fsize = os.path.getsize(csv_path)
        t0 = time.time()
        try:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            safe_path = csv_path.replace('\\', '/')
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{safe_path}')")
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            elapsed = time.time() - t0
            print(f"  {table_name}: {count} rows, {fsize/1024/1024:.1f}MB, {elapsed:.1f}s")
        except Exception as e:
            print(f"  {table_name}: FAILED - {e}")
    conn.close()


def migrate_stock_dailies(db_path: str, data_dir: str, dry_run: bool = False):
    """Import per-stock daily data (data_all_stocks/*/daily.csv) into DuckDB."""
    if not os.path.isdir(data_dir):
        print(f"Stock data directory not found: {data_dir}")
        return

    stock_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found {len(stock_dirs)} stock directories in {data_dir}")

    if dry_run:
        total_csvs = sum(
            1 for d in stock_dirs
            if os.path.exists(os.path.join(data_dir, d, 'daily.csv'))
        )
        print(f"[DRY RUN] Would import {total_csvs} daily.csv files")
        return

    conn = duckdb.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS daily")
    first = True
    imported = 0
    for d in stock_dirs:
        fp = os.path.join(data_dir, d, 'daily.csv')
        if not os.path.exists(fp) or os.path.getsize(fp) < 1024:
            continue
        safe_path = fp.replace('\\', '/')
        try:
            if first:
                conn.execute(f"CREATE TABLE daily AS SELECT * FROM read_csv_auto('{safe_path}')")
                first = False
            else:
                conn.execute(f"INSERT INTO daily SELECT * FROM read_csv_auto('{safe_path}')")
            imported += 1
            if imported % 500 == 0:
                print(f"  Imported {imported}/{len(stock_dirs)} stocks...")
        except Exception as e:
            if imported < 5:
                print(f"  {d}: FAILED - {e}")
    total = conn.execute("SELECT COUNT(*) FROM daily").fetchone()[0]
    print(f"  daily: {total} rows from {imported} stocks")
    conn.close()


def main():
    dry_run = '--dry-run' in sys.argv
    db_path = os.path.join(os.path.dirname(LOCAL_GLOBAL_DIR), 'quant.duckdb')
    print(f"DuckDB path: {db_path}")
    print("=" * 50)

    print("\n[1/2] Global CSVs...")
    migrate_global_csvs(db_path, LOCAL_GLOBAL_DIR, dry_run)

    print("\n[2/2] Stock daily data...")
    migrate_stock_dailies(db_path, LOCAL_DATA_DIR, dry_run)

    if not dry_run:
        print(f"\nDone. Database: {db_path}")
        conn = duckdb.connect(db_path)
        tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()
        print(f"Tables ({len(tables)}): {', '.join(t[0] for t in tables)}")
        conn.close()


if __name__ == '__main__':
    main()
