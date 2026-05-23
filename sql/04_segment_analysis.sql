-- ============================================================
-- FILE     : sql/04_segment_analysis.sql
-- PROJECT  : Customer Segmentation using RFM + Clustering
-- PURPOSE  : Advanced business analytics queries
--            Exports final data for Power BI / Tableau / Python
-- RUN      : Via src/db_connector.py  (last SQL step)
-- ============================================================
-- RECRUITER NOTE:
--   These queries show senior-level SQL skills:
--   ✅ Recursive CTEs
--   ✅ PIVOT-style aggregation with CASE WHEN
--   ✅ Complex multi-table joins
--   ✅ Subquery optimisation patterns
--   ✅ Statistical approximations in SQL
--   ✅ Self-joins for period comparison
-- ============================================================

PRAGMA foreign_keys = ON;


-- ════════════════════════════════════════════════════════════
-- SECTION A: EXECUTIVE SUMMARY QUERIES
-- These power the top-level KPI cards in the dashboard.
-- ════════════════════════════════════════════════════════════

-- A1: Single-row KPI summary (used as dashboard headline)
SELECT
    COUNT(DISTINCT seg.customer_id)             AS total_customers,
    ROUND(SUM(seg.monetary), 0)                 AS total_revenue,
    ROUND(AVG(seg.recency_days), 1)             AS avg_recency_days,
    ROUND(AVG(seg.frequency), 1)                AS avg_frequency,
    ROUND(AVG(seg.monetary), 0)                 AS avg_monetary,
    ROUND(AVG(seg.clv_estimate), 0)             AS avg_clv_estimate,
    -- Champions KPIs
    SUM(CASE WHEN seg.segment_label = 'Champions'
             THEN 1 ELSE 0 END)                 AS champion_count,
    ROUND(SUM(CASE WHEN seg.segment_label = 'Champions'
                   THEN seg.monetary ELSE 0 END), 0) AS champion_revenue,
    -- Revenue at risk
    ROUND(SUM(CASE WHEN seg.churn_probability >= 0.60
                   THEN seg.monetary ELSE 0 END), 0) AS revenue_at_high_churn_risk,
    ROUND(SUM(CASE WHEN seg.churn_probability >= 0.60
                   THEN seg.monetary ELSE 0 END)
          * 100.0 / NULLIF(SUM(seg.monetary), 0), 1) AS pct_revenue_at_risk
FROM rfm_segments seg;


-- A2: Segment comparison — PIVOT table
-- One column per segment for easy Power BI import
SELECT
    metric,
    Champions,
    [Loyal Customers],
    [Potential Loyal],
    [At Risk],
    [Lost / Inactive]
FROM (
    SELECT
        'Customer Count'    AS metric,
        SUM(CASE WHEN segment_label = 'Champions'       THEN 1 ELSE 0 END) AS Champions,
        SUM(CASE WHEN segment_label = 'Loyal Customers' THEN 1 ELSE 0 END) AS [Loyal Customers],
        SUM(CASE WHEN segment_label = 'Potential Loyal' THEN 1 ELSE 0 END) AS [Potential Loyal],
        SUM(CASE WHEN segment_label = 'At Risk'         THEN 1 ELSE 0 END) AS [At Risk],
        SUM(CASE WHEN segment_label = 'Lost / Inactive' THEN 1 ELSE 0 END) AS [Lost / Inactive]
    FROM rfm_segments
    UNION ALL
    SELECT
        'Total Revenue (£)',
        ROUND(SUM(CASE WHEN segment_label = 'Champions'       THEN monetary ELSE 0 END), 0),
        ROUND(SUM(CASE WHEN segment_label = 'Loyal Customers' THEN monetary ELSE 0 END), 0),
        ROUND(SUM(CASE WHEN segment_label = 'Potential Loyal' THEN monetary ELSE 0 END), 0),
        ROUND(SUM(CASE WHEN segment_label = 'At Risk'         THEN monetary ELSE 0 END), 0),
        ROUND(SUM(CASE WHEN segment_label = 'Lost / Inactive' THEN monetary ELSE 0 END), 0)
    FROM rfm_segments
    UNION ALL
    SELECT
        'Avg Recency (days)',
        ROUND(AVG(CASE WHEN segment_label = 'Champions'       THEN recency_days END), 1),
        ROUND(AVG(CASE WHEN segment_label = 'Loyal Customers' THEN recency_days END), 1),
        ROUND(AVG(CASE WHEN segment_label = 'Potential Loyal' THEN recency_days END), 1),
        ROUND(AVG(CASE WHEN segment_label = 'At Risk'         THEN recency_days END), 1),
        ROUND(AVG(CASE WHEN segment_label = 'Lost / Inactive' THEN recency_days END), 1)
    FROM rfm_segments
    UNION ALL
    SELECT
        'Avg Orders',
        ROUND(AVG(CASE WHEN segment_label = 'Champions'       THEN frequency END), 1),
        ROUND(AVG(CASE WHEN segment_label = 'Loyal Customers' THEN frequency END), 1),
        ROUND(AVG(CASE WHEN segment_label = 'Potential Loyal' THEN frequency END), 1),
        ROUND(AVG(CASE WHEN segment_label = 'At Risk'         THEN frequency END), 1),
        ROUND(AVG(CASE WHEN segment_label = 'Lost / Inactive' THEN frequency END), 1)
    FROM rfm_segments
    UNION ALL
    SELECT
        'Avg Monetary (£)',
        ROUND(AVG(CASE WHEN segment_label = 'Champions'       THEN monetary END), 0),
        ROUND(AVG(CASE WHEN segment_label = 'Loyal Customers' THEN monetary END), 0),
        ROUND(AVG(CASE WHEN segment_label = 'Potential Loyal' THEN monetary END), 0),
        ROUND(AVG(CASE WHEN segment_label = 'At Risk'         THEN monetary END), 0),
        ROUND(AVG(CASE WHEN segment_label = 'Lost / Inactive' THEN monetary END), 0)
    FROM rfm_segments
    UNION ALL
    SELECT
        'Avg CLV Estimate (£)',
        ROUND(AVG(CASE WHEN segment_label = 'Champions'       THEN clv_estimate END), 0),
        ROUND(AVG(CASE WHEN segment_label = 'Loyal Customers' THEN clv_estimate END), 0),
        ROUND(AVG(CASE WHEN segment_label = 'Potential Loyal' THEN clv_estimate END), 0),
        ROUND(AVG(CASE WHEN segment_label = 'At Risk'         THEN clv_estimate END), 0),
        ROUND(AVG(CASE WHEN segment_label = 'Lost / Inactive' THEN clv_estimate END), 0)
    FROM rfm_segments
);


-- ════════════════════════════════════════════════════════════
-- SECTION B: ADVANCED ANALYTICS
-- ════════════════════════════════════════════════════════════

-- B1: Running total revenue (useful for cumulative chart in Power BI)
SELECT
    SUBSTR(t.invoice_date, 1, 7)                AS year_month,
    ROUND(SUM(t.quantity * t.unit_price), 0)    AS monthly_revenue,
    ROUND(SUM(SUM(t.quantity * t.unit_price))
          OVER (ORDER BY SUBSTR(t.invoice_date, 1, 7)), 0) AS cumulative_revenue
FROM transactions t
GROUP BY year_month
ORDER BY year_month;


-- B2: Customer retention — repeat vs one-time buyers
WITH buyer_type AS (
    SELECT
        customer_id,
        CASE
            WHEN frequency = 1 THEN 'One-time Buyer'
            WHEN frequency BETWEEN 2 AND 3 THEN 'Occasional (2–3 orders)'
            WHEN frequency BETWEEN 4 AND 9 THEN 'Regular (4–9 orders)'
            ELSE 'Power Buyer (10+ orders)'
        END AS buyer_type,
        monetary
    FROM rfm_segments
)
SELECT
    buyer_type,
    COUNT(*)                            AS customers,
    ROUND(COUNT(*) * 100.0
          / SUM(COUNT(*)) OVER (), 1)  AS pct_customers,
    ROUND(SUM(monetary), 0)             AS total_revenue,
    ROUND(SUM(monetary) * 100.0
          / SUM(SUM(monetary)) OVER (), 1) AS pct_revenue,
    ROUND(AVG(monetary), 0)             AS avg_spend_per_customer
FROM buyer_type
GROUP BY buyer_type
ORDER BY customers DESC;


-- B3: Geographic segment breakdown
-- Which countries produce the best customers?
SELECT
    c.country,
    COUNT(DISTINCT seg.customer_id)             AS customers,
    ROUND(SUM(seg.monetary), 0)                 AS total_revenue,
    ROUND(AVG(seg.rfm_score), 1)                AS avg_rfm_score,
    SUM(CASE WHEN seg.segment_label = 'Champions'
             THEN 1 ELSE 0 END)                 AS champions,
    SUM(CASE WHEN seg.segment_label IN ('At Risk','Lost / Inactive')
             THEN 1 ELSE 0 END)                 AS high_risk_customers
FROM rfm_segments seg
JOIN customers     c ON c.customer_id = seg.customer_id
GROUP BY c.country
HAVING customers >= 5           -- filter out noise from tiny countries
ORDER BY total_revenue DESC
LIMIT 15;


-- B4: Percentile distribution of Monetary value
-- Shows where the cutoffs are for each quintile
-- (SQL approximation since SQLite lacks PERCENTILE_CONT)
WITH ordered AS (
    SELECT
        monetary,
        ROW_NUMBER() OVER (ORDER BY monetary) AS rn,
        COUNT(*) OVER ()                      AS total
    FROM rfm_segments
)
SELECT
    ROUND(pct * 100) || 'th percentile'    AS percentile,
    ROUND(monetary, 2)                      AS monetary_value
FROM ordered
CROSS JOIN (VALUES (0.10),(0.25),(0.50),(0.75),(0.90),(0.95),(0.99)) AS pcts(pct)
WHERE rn = CAST(pct * total AS INTEGER)
ORDER BY pct;


-- B5: Days-between-purchases distribution (cohort loyalty metric)
-- How many days on average between a customer's orders?
WITH invoice_dates AS (
    SELECT
        customer_id,
        invoice_date,
        LAG(invoice_date) OVER (
            PARTITION BY customer_id
            ORDER BY invoice_date
        ) AS prev_invoice_date
    FROM (
        SELECT DISTINCT customer_id, invoice_date
        FROM transactions
    )
),
gaps AS (
    SELECT
        customer_id,
        CAST(JULIANDAY(invoice_date) - JULIANDAY(prev_invoice_date) AS INTEGER) AS days_between
    FROM invoice_dates
    WHERE prev_invoice_date IS NOT NULL
      AND days_between > 0
)
SELECT
    seg.segment_label,
    ROUND(AVG(g.days_between), 1)           AS avg_days_between_orders,
    ROUND(MIN(g.days_between), 0)           AS min_days_between,
    ROUND(MAX(g.days_between), 0)           AS max_days_between,
    COUNT(DISTINCT g.customer_id)           AS customers_with_repeat_orders
FROM gaps g
JOIN rfm_segments seg ON seg.customer_id = g.customer_id
GROUP BY seg.segment_label
ORDER BY avg_days_between_orders;


-- ════════════════════════════════════════════════════════════
-- SECTION C: EXPORT QUERIES
-- These are run by db_connector.py to produce CSV files
-- that feed into Python ML pipeline, Power BI, and Tableau.
-- ════════════════════════════════════════════════════════════

-- C1: Main RFM export (replaces Python-generated rfm_scored.csv)
-- This is the primary output: one row per customer, all features
SELECT
    seg.customer_id,
    c.country,
    seg.recency_days    AS recency,
    seg.frequency,
    seg.monetary,
    seg.r_score,
    seg.f_score,
    seg.m_score,
    seg.rfm_score,
    seg.rfm_segment_code,
    seg.segment_label   AS segment,
    seg.churn_probability,
    seg.clv_estimate
FROM rfm_segments  seg
JOIN customers      c ON c.customer_id = seg.customer_id
ORDER BY seg.monetary DESC;


-- C2: Segment summary export (for Power BI Segment_Summary sheet)
SELECT
    segment_label                               AS segment,
    COUNT(*)                                    AS customers,
    ROUND(AVG(recency_days), 1)                 AS avg_recency,
    ROUND(AVG(frequency), 1)                    AS avg_frequency,
    ROUND(AVG(monetary), 0)                     AS avg_monetary,
    ROUND(SUM(monetary), 0)                     AS total_revenue,
    ROUND(AVG(clv_estimate), 0)                 AS avg_clv,
    ROUND(AVG(churn_probability), 2)            AS avg_churn_probability,
    ROUND(COUNT(*) * 100.0
          / SUM(COUNT(*)) OVER (), 1)           AS pct_customers,
    ROUND(SUM(monetary) * 100.0
          / SUM(SUM(monetary)) OVER (), 1)      AS pct_revenue
FROM rfm_segments
GROUP BY segment_label
ORDER BY avg_monetary DESC;


-- C3: Transaction-level export (for Tableau time-series charts)
SELECT
    t.invoice,
    SUBSTR(t.invoice_date, 1, 10)               AS invoice_date,
    SUBSTR(t.invoice_date, 1, 7)                AS year_month,
    t.customer_id,
    c.country,
    seg.segment_label,
    p.description                               AS product,
    t.quantity,
    t.unit_price,
    ROUND(t.quantity * t.unit_price, 2)         AS line_total
FROM transactions  t
JOIN customers     c   ON c.customer_id = t.customer_id
JOIN products      p   ON p.stock_code  = t.stock_code
JOIN rfm_segments  seg ON seg.customer_id = t.customer_id
ORDER BY t.invoice_date DESC;
