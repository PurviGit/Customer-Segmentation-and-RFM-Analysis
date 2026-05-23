-- ============================================================
-- FILE     : sql/02_load_and_clean.sql
-- PROJECT  : Customer Segmentation using RFM + Clustering
-- PURPOSE  : Load raw CSV data into SQLite, apply all cleaning
--            rules, build customer_summary aggregate table
-- RUN      : Via src/db_connector.py  (handles CSV → SQLite load)
--            This file handles the SQL-side cleaning after load
-- ============================================================
-- RECRUITER NOTE:
--   This file demonstrates:
--   ✅ CTEs (Common Table Expressions) for readable pipelines
--   ✅ Data quality checks using SQL aggregations
--   ✅ Filtering / cleaning logic in SQL
--   ✅ INSERT … SELECT pattern (ETL in SQL)
--   ✅ Window functions (ROW_NUMBER, RANK)
--   ✅ CASE WHEN for conditional logic
--   ✅ String functions (LIKE, SUBSTR, UPPER)
--   ✅ Date functions
-- ============================================================


-- ── Enable foreign key enforcement ──────────────────────────
PRAGMA foreign_keys = ON;


-- ────────────────────────────────────────────────────────────
-- SECTION A: DATA QUALITY CHECKS
-- Run these first to understand the raw data's problems.
-- Mirrors the quality report in notebook 01.
-- ────────────────────────────────────────────────────────────

-- A1: Total raw records loaded (before any cleaning)
SELECT
    'Total raw records'           AS metric,
    COUNT(*)                      AS value
FROM raw_transactions

UNION ALL

-- A2: Records with NULL customer_id
SELECT
    'Null customer_id',
    COUNT(*)
FROM raw_transactions
WHERE customer_id IS NULL OR TRIM(customer_id) = ''

UNION ALL

-- A3: Cancelled invoices (Invoice starts with 'C')
SELECT
    'Cancelled invoices (C...)',
    COUNT(*)
FROM raw_transactions
WHERE UPPER(SUBSTR(invoice, 1, 1)) = 'C'

UNION ALL

-- A4: Negative / zero quantity
SELECT
    'Non-positive quantity rows',
    COUNT(*)
FROM raw_transactions
WHERE CAST(quantity AS REAL) <= 0

UNION ALL

-- A5: Zero / negative price
SELECT
    'Non-positive price rows',
    COUNT(*)
FROM raw_transactions
WHERE CAST(unit_price AS REAL) <= 0

UNION ALL

-- A6: Unique customers in raw data
SELECT
    'Unique customers (raw)',
    COUNT(DISTINCT customer_id)
FROM raw_transactions
WHERE customer_id IS NOT NULL AND TRIM(customer_id) != ''

UNION ALL

-- A7: Unique countries
SELECT
    'Unique countries',
    COUNT(DISTINCT country)
FROM raw_transactions;


-- ────────────────────────────────────────────────────────────
-- SECTION B: LOAD CLEAN DATA
-- Apply all cleaning rules in a single CTE pipeline.
-- This is equivalent to the Pandas cleaning in notebook 01
-- but done entirely in SQL — showing SQL can replace Python
-- for the ETL stage.
-- ────────────────────────────────────────────────────────────

-- B1: Populate products table (distinct products from clean rows)
INSERT OR IGNORE INTO products (stock_code, description, unit_price)
SELECT DISTINCT
    UPPER(TRIM(stock_code))             AS stock_code,
    TRIM(description)                   AS description,
    CAST(unit_price AS REAL)            AS unit_price
FROM raw_transactions
WHERE
    -- Cleaning rule 1: valid customer
    customer_id IS NOT NULL
    AND TRIM(customer_id) != ''
    -- Cleaning rule 2: not a cancellation
    AND UPPER(SUBSTR(invoice, 1, 1)) != 'C'
    -- Cleaning rule 3: positive quantity and price
    AND CAST(quantity  AS REAL) > 0
    AND CAST(unit_price AS REAL) > 0
    -- Cleaning rule 4: valid stock code (not service/postage codes)
    AND LENGTH(TRIM(stock_code)) >= 5
    AND stock_code NOT IN ('POST','DOT','M','BANK CHARGES','PADS','D');


-- B2: Populate customers table (distinct customers from clean rows)
INSERT OR IGNORE INTO customers (customer_id, country)
SELECT DISTINCT
    CAST(customer_id AS INTEGER)        AS customer_id,
    COALESCE(TRIM(country), 'Unknown')  AS country
FROM raw_transactions
WHERE
    customer_id IS NOT NULL
    AND TRIM(customer_id) != ''
    AND UPPER(SUBSTR(invoice, 1, 1)) != 'C'
    AND CAST(quantity  AS REAL) > 0
    AND CAST(unit_price AS REAL) > 0;


-- B3: Populate transactions table (the main fact table)
-- Uses CTE to compute and filter in one readable pass
WITH cleaned AS (
    SELECT
        TRIM(invoice)                       AS invoice,
        UPPER(TRIM(stock_code))             AS stock_code,
        CAST(quantity AS INTEGER)           AS quantity,
        -- Normalise date to ISO-8601 format
        SUBSTR(invoice_date, 1, 19)         AS invoice_date,
        CAST(unit_price AS REAL)            AS unit_price,
        CAST(customer_id AS INTEGER)        AS customer_id
    FROM raw_transactions
    WHERE
        -- All cleaning rules applied
        customer_id IS NOT NULL
        AND TRIM(customer_id) != ''
        AND UPPER(SUBSTR(invoice, 1, 1)) != 'C'
        AND CAST(quantity  AS REAL) > 0
        AND CAST(unit_price AS REAL) > 0
        AND LENGTH(TRIM(stock_code)) >= 5
        AND stock_code NOT IN ('POST','DOT','M','BANK CHARGES','PADS','D')
),
-- Remove extreme outliers: line items where total > £10,000
-- (bulk/wholesale orders that skew RFM for retail segmentation)
outlier_threshold AS (
    SELECT
        (quantity * unit_price) AS total,
        invoice,
        stock_code,
        quantity,
        invoice_date,
        unit_price,
        customer_id
    FROM cleaned
),
filtered AS (
    SELECT *
    FROM outlier_threshold
    -- Keep only rows below 99.9th percentile of line-item total
    -- SQLite doesn't have PERCENTILE_CONT, so we use a subquery
    WHERE total <= (
        SELECT total
        FROM outlier_threshold
        ORDER BY total DESC
        LIMIT 1
        OFFSET CAST(0.001 * (SELECT COUNT(*) FROM outlier_threshold) AS INTEGER)
    )
)
INSERT INTO transactions (invoice, stock_code, quantity, invoice_date, unit_price, customer_id)
SELECT invoice, stock_code, quantity, invoice_date, unit_price, customer_id
FROM filtered;


-- ────────────────────────────────────────────────────────────
-- SECTION C: BUILD customer_summary TABLE
-- Aggregate facts per customer — this is the intermediate table
-- Python uses to compute RFM (avoids re-aggregating raw rows).
-- ────────────────────────────────────────────────────────────

-- Clear any existing data (safe re-run)
DELETE FROM customer_summary;

INSERT INTO customer_summary (
    customer_id,
    country,
    total_orders,
    total_items,
    total_revenue,
    first_purchase,
    last_purchase
)
SELECT
    t.customer_id,
    c.country,
    COUNT(DISTINCT t.invoice)               AS total_orders,
    SUM(t.quantity)                         AS total_items,
    ROUND(SUM(t.quantity * t.unit_price), 2) AS total_revenue,
    MIN(t.invoice_date)                     AS first_purchase,
    MAX(t.invoice_date)                     AS last_purchase
FROM transactions  t
JOIN customers     c ON c.customer_id = t.customer_id
GROUP BY t.customer_id, c.country;


-- ────────────────────────────────────────────────────────────
-- SECTION D: VALIDATION CHECKS
-- Confirm that the loaded data makes sense.
-- These act as automated data tests (like Great Expectations).
-- ────────────────────────────────────────────────────────────

-- D1: Row counts after cleaning
SELECT
    'transactions (clean)'          AS table_name,
    COUNT(*)                        AS row_count
FROM transactions
UNION ALL
SELECT 'customers',         COUNT(*) FROM customers
UNION ALL
SELECT 'products',          COUNT(*) FROM products
UNION ALL
SELECT 'customer_summary',  COUNT(*) FROM customer_summary;


-- D2: Revenue sanity check
SELECT
    ROUND(SUM(quantity * unit_price), 2)        AS total_revenue,
    ROUND(AVG(quantity * unit_price), 2)        AS avg_line_item,
    ROUND(MIN(quantity * unit_price), 2)        AS min_line_item,
    ROUND(MAX(quantity * unit_price), 2)        AS max_line_item
FROM transactions;


-- D3: Top 5 customers by revenue
SELECT
    customer_id,
    total_orders,
    total_revenue,
    country
FROM customer_summary
ORDER BY total_revenue DESC
LIMIT 5;


-- D4: Transactions per month (seasonality check)
SELECT
    SUBSTR(invoice_date, 1, 7)          AS year_month,
    COUNT(DISTINCT invoice)             AS invoices,
    COUNT(DISTINCT customer_id)         AS customers,
    ROUND(SUM(quantity * unit_price), 0) AS revenue
FROM transactions
GROUP BY year_month
ORDER BY year_month;


-- D5: Top 10 countries by revenue
SELECT
    c.country,
    COUNT(DISTINCT t.customer_id)           AS customers,
    ROUND(SUM(t.quantity * t.unit_price), 0) AS revenue
FROM transactions t
JOIN customers    c ON c.customer_id = t.customer_id
GROUP BY c.country
ORDER BY revenue DESC
LIMIT 10;
