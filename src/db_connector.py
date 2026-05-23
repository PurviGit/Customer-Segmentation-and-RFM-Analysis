# ============================================================
# FILE     : src/db_connector.py
# PROJECT  : Customer Segmentation using RFM + Clustering
# PURPOSE  : The bridge between SQL and Python ML pipeline
#
#   This script does 5 things:
#   1. Creates the SQLite database (runs 01_create_schema.sql)
#   2. Loads raw CSV/Excel data into a staging table
#   3. Runs cleaning SQL (02_load_and_clean.sql)
#   4. Runs RFM computation SQL (03_rfm_queries.sql)
#   5. Runs analysis SQL (04_segment_analysis.sql)
#   6. Exports the SQL output to CSV for downstream Python notebooks
#
# HOW TO RUN:
#   cd rfm-customer-segmentation
#   python src/db_connector.py
#
# WHAT IT PRODUCES:
#   data/processed/retail.db        ← SQLite database (all tables)
#   data/processed/rfm_from_sql.csv ← RFM table from SQL (feeds ML)
#   data/processed/sql_segment_summary.csv ← Segment summary
#   data/processed/sql_transactions.csv    ← Transaction export
#
# RECRUITER NOTE:
#   Demonstrates:
#   ✅ Python–SQL integration (sqlite3, pandas read_sql)
#   ✅ ETL pipeline design
#   ✅ Error handling and logging
#   ✅ Modular, production-ready code structure
#   ✅ pandas to_sql for loading DataFrames into SQL
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import os
import sys
import logging
from pathlib import Path

# ── Logging setup ────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s  %(levelname)s  %(message)s',
    datefmt = '%H:%M:%S'
)
log = logging.getLogger(__name__)

# ── Path constants ───────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
SQL_DIR    = BASE_DIR / 'sql'
DATA_RAW   = BASE_DIR / 'data' / 'raw'
DATA_PROC  = BASE_DIR / 'data' / 'processed'
DB_PATH    = DATA_PROC / 'retail.db'

DATA_PROC.mkdir(parents=True, exist_ok=True)

RAW_XLSX   = DATA_RAW  / 'online_retail_II.xlsx'
RAW_CSV    = DATA_RAW  / 'online_retail_II.csv'   # alternative input


# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

def get_connection() -> sqlite3.Connection:
    """
    Returns a connection to the SQLite database.
    PRAGMA foreign_keys = ON ensures referential integrity.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")   # faster writes
    return conn


def run_sql_file(conn: sqlite3.Connection, filepath: Path,
                 label: str = "") -> None:
    """
    Reads a .sql file and executes it against the connection.
    Splits on ';' to execute statement by statement.
    Skips empty statements and comment-only blocks.
    """
    log.info(f"Running SQL: {filepath.name}  {label}")
    sql_text = filepath.read_text(encoding='utf-8')

    # Split into individual statements
    statements = [s.strip() for s in sql_text.split(';')]
    executed   = 0

    for stmt in statements:
        # Skip empty or comment-only statements
        lines = [l for l in stmt.splitlines()
                 if l.strip() and not l.strip().startswith('--')]
        if not lines:
            continue
        try:
            conn.execute(stmt)
            executed += 1
        except sqlite3.Error as e:
            # Log warning but continue — some statements may fail
            # (e.g. DROP IF NOT EXISTS on first run)
            log.debug(f"  SQL stmt skipped: {e}  |  stmt[:60]: {stmt[:60]}")

    conn.commit()
    log.info(f"  ✅ {executed} statements executed from {filepath.name}")


def load_raw_data(conn: sqlite3.Connection) -> int:
    """
    Loads raw transaction data from Excel or CSV into a
    SQLite staging table called 'raw_transactions'.

    If the real dataset is not available, generates synthetic
    data automatically so the pipeline always runs.

    Returns: number of rows loaded
    """
    log.info("Loading raw data …")

    # ── Try to load real data ────────────────────────────────
    df = None

    if RAW_XLSX.exists():
        log.info(f"  Found Excel file: {RAW_XLSX.name}")
        df = pd.read_excel(
            RAW_XLSX,
            sheet_name = 'Year 2010-2011',
            dtype      = {'Customer ID': str, 'Invoice': str, 'StockCode': str}
        )
        log.info(f"  Loaded {len(df):,} rows from Excel")

    elif RAW_CSV.exists():
        log.info(f"  Found CSV file: {RAW_CSV.name}")
        df = pd.read_csv(
            RAW_CSV,
            dtype = {'Customer ID': str, 'Invoice': str, 'StockCode': str}
        )
        log.info(f"  Loaded {len(df):,} rows from CSV")

    else:
        log.warning("  ⚠️  Real dataset not found — generating synthetic data")
        log.warning("  Download from: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II")
        df = _generate_synthetic_data(n=50_000)
        log.info(f"  Generated {len(df):,} synthetic rows")

    # ── Standardise column names ─────────────────────────────
    col_map = {
        'Invoice'    : 'invoice',
        'StockCode'  : 'stock_code',
        'Description': 'description',
        'Quantity'   : 'quantity',
        'InvoiceDate': 'invoice_date',
        'Price'      : 'unit_price',
        'Customer ID': 'customer_id',
        'Country'    : 'country',
    }
    df.rename(columns=col_map, inplace=True)

    # Convert invoice_date to string for SQLite storage
    if pd.api.types.is_datetime64_any_dtype(df['invoice_date']):
        df['invoice_date'] = df['invoice_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
        df['invoice_date'] = df['invoice_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # ── Write to SQLite staging table ────────────────────────
    df.to_sql(
    name       = 'raw_transactions',
    con        = conn,
    if_exists  = 'replace',
    index      = False,
    chunksize  = 500
)
    conn.commit()
    log.info(f"  ✅ {len(df):,} rows written to raw_transactions table")
    return len(df)


def _generate_synthetic_data(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """Generates realistic synthetic retail data mirroring UCI schema."""
    np.random.seed(seed)

    rng          = pd.date_range('2010-12-01', '2011-12-09', freq='H')
    n_customers  = 4000
    customer_ids = np.random.randint(12346, 18000, n_customers).astype(str)
    weights      = np.random.pareto(1.5, n_customers) + 0.1
    weights     /= weights.sum()

    products = {f"{''.join(np.random.choice(list('ABCDEF'), 5))}":
                np.random.choice(['WHITE METAL LANTERN','CREAM CUPID HEARTS COAT HANGER',
                                  'GLASS STAR FROSTED T-LIGHT HOLDER','RED WOOLLY HOTTIE',
                                  'KNITTED UNION FLAG HOT WATER BOTTLE'])
                for _ in range(200)}

    stock_codes  = list(products.keys())
    descriptions = list(products.values())

    df = pd.DataFrame({
        'Invoice'    : [f"{'C' if np.random.rand() < 0.02 else ''}{500000+i}" for i in range(n)],
        'StockCode'  : np.random.choice(stock_codes, n),
        'Description': np.random.choice(descriptions, n),
        'Quantity'   : np.random.choice(list(range(-3,0)) + list(range(1,40)),
                                         n, p=[0.01]*3 + [0.97/39]*39),
        'InvoiceDate': pd.to_datetime(np.random.choice(rng, n)),
        'Price'      : np.round(np.random.lognormal(1.5, 0.8, n), 2).clip(0.01, 200),
        'Customer ID': np.random.choice(customer_ids, n, p=weights),
        'Country'    : np.random.choice(
                           ['United Kingdom','Germany','France','Netherlands','Australia'],
                           n, p=[0.82, 0.06, 0.06, 0.03, 0.03]),
    })
    # Inject 8% null Customer IDs
    null_mask = np.random.rand(n) < 0.08
    df.loc[null_mask, 'Customer ID'] = np.nan
    return df


def export_sql_outputs(conn: sqlite3.Connection) -> None:
    """
    Reads the final SQL tables and exports them to CSV files.
    These CSVs are then used by:
      - Python notebooks (ML pipeline)
      - Power BI (via powerbi_data.xlsx update)
      - Tableau
      - Streamlit dashboard
    """
    log.info("Exporting SQL outputs to CSV …")

    exports = {
        'rfm_from_sql.csv': """
            SELECT
                seg.customer_id,
                c.country,
                seg.recency_days        AS recency,
                seg.frequency,
                seg.monetary,
                seg.r_score,
                seg.f_score,
                seg.m_score,
                seg.rfm_score,
                seg.rfm_segment_code,
                seg.segment_label       AS segment,
                seg.churn_probability,
                seg.clv_estimate
            FROM rfm_segments seg
            JOIN customers    c ON c.customer_id = seg.customer_id
            ORDER BY seg.monetary DESC
        """,
        'sql_segment_summary.csv': """
            SELECT
                segment_label                                   AS segment,
                COUNT(*)                                        AS customers,
                ROUND(AVG(recency_days), 1)                     AS avg_recency,
                ROUND(AVG(frequency), 1)                        AS avg_frequency,
                ROUND(AVG(monetary), 0)                         AS avg_monetary,
                ROUND(SUM(monetary), 0)                         AS total_revenue,
                ROUND(AVG(clv_estimate), 0)                     AS avg_clv,
                ROUND(AVG(churn_probability) * 100, 1)          AS avg_churn_pct
            FROM rfm_segments
            GROUP BY segment_label
            ORDER BY avg_monetary DESC
        """,
        'sql_kpi_summary.csv': """
            SELECT
                COUNT(DISTINCT customer_id)             AS total_customers,
                ROUND(SUM(monetary), 0)                 AS total_revenue,
                ROUND(AVG(recency_days), 1)             AS avg_recency_days,
                ROUND(AVG(frequency), 1)                AS avg_frequency,
                ROUND(AVG(monetary), 0)                 AS avg_monetary,
                ROUND(AVG(clv_estimate), 0)             AS avg_clv,
                ROUND(SUM(CASE WHEN churn_probability >= 0.60
                               THEN monetary ELSE 0 END), 0) AS revenue_at_risk
            FROM rfm_segments
        """,
    }

    for filename, query in exports.items():
        try:
            df_out = pd.read_sql_query(query, conn)
            out_path = DATA_PROC / filename
            df_out.to_csv(out_path, index=False)
            log.info(f"  ✅ Exported {len(df_out):,} rows → {filename}")
        except Exception as e:
            log.error(f"  ❌ Failed to export {filename}: {e}")


def run_analytical_queries(conn: sqlite3.Connection) -> None:
    """
    Runs the key analytical queries from 03_rfm_queries.sql
    and prints results to console. Mirrors what you'd see in
    DBeaver / pgAdmin / any SQL client.
    """
    log.info("Running analytical queries …")

    queries = {
        "Customers per Segment": """
            SELECT segment_label, COUNT(*) AS customers,
                   ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(),1) AS pct
            FROM rfm_segments GROUP BY segment_label
            ORDER BY customers DESC
        """,
        "Revenue per Segment": """
            SELECT segment_label,
                   ROUND(SUM(monetary),0) AS total_revenue,
                   ROUND(AVG(monetary),0) AS avg_monetary
            FROM rfm_segments GROUP BY segment_label
            ORDER BY total_revenue DESC
        """,
        "Churn Risk Summary": """
            SELECT CASE WHEN churn_probability>=0.80 THEN 'Critical'
                        WHEN churn_probability>=0.55 THEN 'High'
                        WHEN churn_probability>=0.20 THEN 'Medium'
                        ELSE 'Low' END AS churn_risk,
                   COUNT(*) AS customers,
                   ROUND(SUM(monetary),0) AS revenue_at_risk
            FROM rfm_segments
            GROUP BY churn_risk ORDER BY revenue_at_risk DESC
        """,
        "Pareto Check (top 20% customers)": """
            WITH ranked AS (
                SELECT monetary,
                       ROW_NUMBER() OVER (ORDER BY monetary DESC) AS rn,
                       COUNT(*) OVER () AS total,
                       SUM(monetary) OVER () AS total_rev
                FROM rfm_segments
            )
            SELECT
                ROUND(rn*100.0/total,0) AS top_pct_customers,
                ROUND(SUM(monetary) OVER (ORDER BY rn)/total_rev*100,1) AS pct_revenue
            FROM ranked
            WHERE ROUND(rn*100.0/total,0) IN (10,20,30,50)
            GROUP BY top_pct_customers
        """,
    }

    for title, q in queries.items():
        print(f"\n{'─'*50}")
        print(f"  {title}")
        print(f"{'─'*50}")
        try:
            df = pd.read_sql_query(q, conn)
            print(df.to_string(index=False))
        except Exception as e:
            print(f"  Error: {e}")


# ════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════

def run_sql_pipeline() -> None:
    """
    Orchestrates the full SQL pipeline in order:
    1. Create schema
    2. Load raw data
    3. Clean + aggregate
    4. RFM computation
    5. Advanced analytics
    6. Export CSV outputs
    """
    log.info("=" * 55)
    log.info("  RFM SQL Pipeline Starting")
    log.info("=" * 55)
    log.info(f"  Database : {DB_PATH}")

    # Remove old database for clean run
    if DB_PATH.exists():
        DB_PATH.unlink()
        log.info("  Removed old database for fresh run")

    conn = get_connection()

    try:
        # ── Step 1: Create schema ────────────────────────────
        log.info("\n[Step 1/6] Creating database schema …")
        run_sql_file(conn, SQL_DIR / '01_create_schema.sql', "(schema)")

        # ── Step 2: Load raw data ────────────────────────────
        log.info("\n[Step 2/6] Loading raw data …")
        n_rows = load_raw_data(conn)

        # ── Step 3: Clean & aggregate ────────────────────────
        log.info("\n[Step 3/6] Running cleaning & aggregation SQL …")
        run_sql_file(conn, SQL_DIR / '02_load_and_clean.sql', "(cleaning)")

        # Verify load
        n_clean = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM transactions", conn
        )['n'].iloc[0]
        log.info(f"  Raw rows: {n_rows:,}  →  Clean rows: {n_clean:,}")

        # ── Step 4: RFM computation ──────────────────────────
        log.info("\n[Step 4/6] Computing RFM scores in SQL …")
        run_sql_file(conn, SQL_DIR / '03_rfm_queries.sql', "(RFM)")

        n_rfm = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM rfm_segments", conn
        )['n'].iloc[0]
        log.info(f"  RFM profiles built: {n_rfm:,} customers")

        # ── Step 5: Advanced analytics ───────────────────────
        log.info("\n[Step 5/6] Running advanced analytics SQL …")
        run_sql_file(conn, SQL_DIR / '04_segment_analysis.sql', "(analysis)")

        # ── Step 6: Export ───────────────────────────────────
        log.info("\n[Step 6/6] Exporting SQL outputs to CSV …")
        export_sql_outputs(conn)

        # Print key results
        run_analytical_queries(conn)

        log.info("\n" + "=" * 55)
        log.info("  ✅ SQL Pipeline Complete!")
        log.info("=" * 55)
        log.info(f"\n  Database  : {DB_PATH}")
        log.info(f"  Main CSV  : {DATA_PROC / 'rfm_from_sql.csv'}")
        log.info(f"  Summary   : {DATA_PROC / 'sql_segment_summary.csv'}")
        log.info("\n  ▶  Next steps:")
        log.info("     1. Run: python notebooks/02_rfm_engineering.py")
        log.info("        (it will read rfm_from_sql.csv automatically)")
        log.info("     2. Run: python notebooks/03_clustering.py")
        log.info("     3. Run: python notebooks/04_insights.py")
        log.info("     4. Run: streamlit run dashboard/app.py")

    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        raise

    finally:
        conn.close()


def query_db(sql: str) -> pd.DataFrame:
    """
    Convenience function: run any SQL query against the database
    and return a pandas DataFrame.

    Usage from Python:
        from src.db_connector import query_db
        df = query_db("SELECT * FROM rfm_segments WHERE segment_label = 'Champions'")
    """
    conn = get_connection()
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


if __name__ == '__main__':
    run_sql_pipeline()
