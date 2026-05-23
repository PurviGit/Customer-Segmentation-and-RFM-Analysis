-- ============================================================
-- FILE     : sql/01_create_schema.sql
-- PROJECT  : Customer Segmentation using RFM + Clustering
-- PURPOSE  : Create the database schema and all tables
-- ENGINE   : SQLite (file-based, no server needed)
--            Compatible with PostgreSQL / MySQL with minor edits
-- RUN      : Automatically via src/db_connector.py
--            OR manually: sqlite3 data/processed/retail.db < sql/01_create_schema.sql
-- ============================================================
-- RECRUITER NOTE:
--   This file demonstrates:
--   ✅ Proper schema design with data types
--   ✅ Primary & foreign key constraints
--   ✅ Indexes for query performance
--   ✅ Use of CHECK constraints for data validation
--   ✅ Comments documenting every design decision
-- ============================================================


-- ── Drop tables if they exist (safe re-run) ─────────────────
-- ORDER MATTERS: drop child tables before parent tables
DROP TABLE IF EXISTS rfm_segments;
DROP TABLE IF EXISTS rfm_scores;
DROP TABLE IF EXISTS customer_summary;
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS products;


-- ── Table 1: products ────────────────────────────────────────
-- Master list of products sold by the retailer.
-- Normalised out of transactions to avoid storing description
-- 100,000 times (one per transaction row).
CREATE TABLE products (
    stock_code      TEXT        PRIMARY KEY,
    description     TEXT        NOT NULL,
    unit_price      REAL        NOT NULL CHECK (unit_price >= 0)
);


-- ── Table 2: customers ───────────────────────────────────────
-- One row per unique customer.
-- customer_id is the natural key from the source system.
CREATE TABLE customers (
    customer_id     INTEGER     PRIMARY KEY,
    country         TEXT        NOT NULL DEFAULT 'Unknown'
);


-- ── Table 3: transactions ────────────────────────────────────
-- Fact table — one row per line item on an invoice.
-- A single invoice (basket) can have many line items.
CREATE TABLE transactions (
    transaction_id  INTEGER     PRIMARY KEY AUTOINCREMENT,
    invoice         TEXT        NOT NULL,
    stock_code      TEXT        NOT NULL,
    quantity        INTEGER     NOT NULL CHECK (quantity > 0),
    invoice_date    TEXT        NOT NULL,   -- stored as ISO-8601: 'YYYY-MM-DD HH:MM:SS'
    unit_price      REAL        NOT NULL CHECK (unit_price > 0),
    customer_id     INTEGER     NOT NULL,
    total_amount    REAL        GENERATED ALWAYS AS (quantity * unit_price) VIRTUAL,
    -- Foreign keys (enforced when PRAGMA foreign_keys = ON)
    FOREIGN KEY (stock_code)   REFERENCES products(stock_code),
    FOREIGN KEY (customer_id)  REFERENCES customers(customer_id)
);

-- Indexes for common query patterns
-- Without these, every GROUP BY customer_id scans the full table
CREATE INDEX IF NOT EXISTS idx_transactions_customer
    ON transactions(customer_id);

CREATE INDEX IF NOT EXISTS idx_transactions_date
    ON transactions(invoice_date);

CREATE INDEX IF NOT EXISTS idx_transactions_invoice
    ON transactions(invoice);


-- ── Table 4: customer_summary ────────────────────────────────
-- Pre-aggregated per-customer metrics.
-- Built by 02_load_and_clean.sql.
-- This is what Python reads for ML (avoids re-aggregating every run).
CREATE TABLE customer_summary (
    customer_id     INTEGER     PRIMARY KEY,
    country         TEXT,
    total_orders    INTEGER     NOT NULL DEFAULT 0,
    total_items     INTEGER     NOT NULL DEFAULT 0,
    total_revenue   REAL        NOT NULL DEFAULT 0.0,
    first_purchase  TEXT,       -- ISO date of earliest invoice
    last_purchase   TEXT,       -- ISO date of most recent invoice
    avg_order_value REAL        GENERATED ALWAYS AS
                        (CASE WHEN total_orders > 0
                              THEN total_revenue / total_orders
                              ELSE 0 END) VIRTUAL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);


-- ── Table 5: rfm_scores ──────────────────────────────────────
-- RFM metrics + quintile scores per customer.
-- Built entirely in SQL by 03_rfm_queries.sql.
-- Demonstrates SQL window functions (NTILE, RANK, etc.)
CREATE TABLE rfm_scores (
    customer_id     INTEGER     PRIMARY KEY,
    recency_days    INTEGER     NOT NULL,   -- days since last purchase
    frequency       INTEGER     NOT NULL,   -- number of unique invoices
    monetary        REAL        NOT NULL,   -- total spend £
    r_score         INTEGER     NOT NULL CHECK (r_score BETWEEN 1 AND 5),
    f_score         INTEGER     NOT NULL CHECK (f_score BETWEEN 1 AND 5),
    m_score         INTEGER     NOT NULL CHECK (m_score BETWEEN 1 AND 5),
    rfm_score       INTEGER     GENERATED ALWAYS AS (r_score + f_score + m_score) VIRTUAL,
    rfm_segment     TEXT        GENERATED ALWAYS AS
                        (CAST(r_score AS TEXT) ||
                         CAST(f_score AS TEXT) ||
                         CAST(m_score AS TEXT)) VIRTUAL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);


-- ── Table 6: rfm_segments ────────────────────────────────────
-- Final business segment labels per customer.
-- Combines rfm_scores with the business naming logic.
-- This is the table exported to CSV for Power BI / Tableau / Streamlit.
CREATE TABLE rfm_segments (
    customer_id         INTEGER     PRIMARY KEY,
    recency_days        INTEGER,
    frequency           INTEGER,
    monetary            REAL,
    r_score             INTEGER,
    f_score             INTEGER,
    m_score             INTEGER,
    rfm_score           INTEGER,
    rfm_segment_code    TEXT,
    segment_label       TEXT        NOT NULL,
    churn_probability   REAL,
    clv_estimate        REAL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE INDEX IF NOT EXISTS idx_rfm_segments_label
    ON rfm_segments(segment_label);


-- ── Verify schema ────────────────────────────────────────────
SELECT
    name        AS table_name,
    sql         AS create_statement
FROM sqlite_master
WHERE type = 'table'
ORDER BY name;
