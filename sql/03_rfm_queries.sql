-- ============================================================
-- FILE     : sql/03_rfm_queries.sql
-- PROJECT  : Customer Segmentation using RFM + Clustering
-- PURPOSE  : Compute full RFM metrics and scores entirely in SQL
--            using CTEs and window functions (NTILE)
-- RUN      : Via src/db_connector.py after 02_load_and_clean.sql
-- ============================================================
-- RECRUITER NOTE:
--   This is the most SQL-skill-intensive file in the project.
--   It demonstrates:
--   ✅ Multi-level CTEs (chained, readable pipeline)
--   ✅ Window functions: NTILE(), RANK(), DENSE_RANK()
--   ✅ CASE WHEN scoring logic
--   ✅ Subquery for snapshot date
--   ✅ JULIANDAY() for date arithmetic
--   ✅ Computed columns (rfm_score, segment_code)
--   ✅ INSERT INTO … SELECT pipeline
--   ✅ Self-documenting query structure
--
--   These are EXACTLY the SQL skills tested in data analyst
--   interviews at companies like Amazon, Flipkart, Zomato,
--   Google, and most analytics-driven firms.
-- ============================================================


-- ── Enable foreign keys ──────────────────────────────────────
PRAGMA foreign_keys = ON;


-- ════════════════════════════════════════════════════════════
-- STEP 1: COMPUTE RFM METRICS
-- One row per customer with Recency, Frequency, Monetary
-- ════════════════════════════════════════════════════════════

-- The "snapshot date" is the day after the last transaction.
-- Think of it as "today" in the dataset's timeline.
-- We subtract each customer's last_purchase from this to get Recency.

WITH snapshot AS (
    -- CTE 1: Determine the reference date
    -- Using a subquery so it's computed once and reused
    SELECT
        DATE(
            MAX(invoice_date), '+1 day'
        )   AS snapshot_date
    FROM transactions
),

rfm_raw AS (
    -- CTE 2: Compute raw RFM values per customer
    SELECT
        cs.customer_id,
        cs.country,

        -- RECENCY: days between last purchase and snapshot
        -- Lower = better (bought recently)
        CAST(
            JULIANDAY((SELECT snapshot_date FROM snapshot))
            - JULIANDAY(cs.last_purchase)
        AS INTEGER)                                         AS recency_days,

        -- FREQUENCY: number of distinct invoices
        -- Higher = better (buys often)
        cs.total_orders                                     AS frequency,

        -- MONETARY: total spend in £
        -- Higher = better (spends more)
        cs.total_revenue                                    AS monetary

    FROM customer_summary cs
    -- Only include customers with at least 1 completed transaction
    WHERE cs.total_orders >= 1
      AND cs.total_revenue > 0
),

rfm_ranked AS (
    -- CTE 3: Rank customers within each metric for NTILE scoring
    -- RANK() breaks ties, needed before NTILE
    SELECT
        *,
        -- For Recency: RANK ascending (smallest recency = best = rank 1)
        RANK() OVER (ORDER BY recency_days ASC)     AS recency_rank,

        -- For Frequency: use DENSE_RANK to handle many ties
        -- (many customers with frequency=1)
        DENSE_RANK() OVER (ORDER BY frequency DESC) AS frequency_rank,

        -- For Monetary: straightforward ranking
        RANK() OVER (ORDER BY monetary DESC)        AS monetary_rank,

        -- Count total customers for NTILE calculation
        COUNT(*) OVER ()                            AS total_customers

    FROM rfm_raw
),

rfm_ntile AS (
    -- CTE 4: Apply NTILE(5) to divide into quintile bands 1–5
    -- NTILE splits the ordered set into N equal-sized buckets
    -- Bucket 1 = bottom 20%, Bucket 5 = top 20%

    SELECT
        customer_id,
        country,
        recency_days,
        frequency,
        monetary,

        -- R Score: High score = LOW recency (bought recently)
        -- NTILE on ASC rank → bucket 1 = most recent → score 5
        -- We REVERSE the ntile bucket to get score (6 - bucket)
        (6 - NTILE(5) OVER (ORDER BY recency_days ASC))    AS r_score,

        -- F Score: High score = HIGH frequency
        NTILE(5) OVER (ORDER BY frequency_rank ASC)        AS f_score,

        -- M Score: High score = HIGH monetary
        NTILE(5) OVER (ORDER BY monetary_rank ASC)         AS m_score

    FROM rfm_ranked
),

rfm_scored AS (
    -- CTE 5: Add composite score and segment code
    SELECT
        customer_id,
        country,
        recency_days,
        frequency,
        monetary,
        r_score,
        f_score,
        m_score,
        (r_score + f_score + m_score)                       AS rfm_total_score,
        (CAST(r_score AS TEXT)
         || CAST(f_score AS TEXT)
         || CAST(m_score AS TEXT))                          AS rfm_segment_code
    FROM rfm_ntile
    -- Ensure scores are in valid range (NTILE can sometimes produce edge values)
    WHERE r_score BETWEEN 1 AND 5
      AND f_score BETWEEN 1 AND 5
      AND m_score BETWEEN 1 AND 5
),

rfm_labelled AS (
    -- CTE 6: Map composite score → business segment label
    -- This is the business interpretation of the ML output
    SELECT
        customer_id,
        country,
        recency_days,
        frequency,
        ROUND(monetary, 2)                                  AS monetary,
        r_score,
        f_score,
        m_score,
        rfm_total_score,
        rfm_segment_code,

        -- Segment label: business-readable name
        CASE
            WHEN rfm_total_score >= 12 THEN 'Champions'
            WHEN rfm_total_score >= 9  THEN 'Loyal Customers'
            WHEN rfm_total_score >= 7  THEN 'Potential Loyal'
            WHEN rfm_total_score >= 5  THEN 'At Risk'
            ELSE                            'Lost / Inactive'
        END                                                 AS segment_label,

        -- Churn probability: estimated from recency
        CASE
            WHEN recency_days > 180 THEN 0.85
            WHEN recency_days > 90  THEN 0.60
            WHEN recency_days > 30  THEN 0.25
            ELSE                         0.05
        END                                                 AS churn_probability,

        -- Estimated 12-month CLV: simple proxy
        -- (annual spend rate based on historical behaviour)
        ROUND(
            (monetary / MAX(recency_days, 1)) * 365
        , 2)                                                AS clv_estimate

    FROM rfm_scored
)

-- ── Final INSERT into rfm_segments table ─────────────────────
-- Clears old data and loads fresh results
-- This is the table Python reads for ML clustering

-- Note: SQLite doesn't support INSERT OR REPLACE with RETURNING,
-- so we DELETE first, then INSERT
;   -- end the CTE block

DELETE FROM rfm_segments;
DELETE FROM rfm_scores;


-- Re-run the full CTE chain to INSERT into rfm_scores
WITH snapshot AS (
    SELECT DATE(MAX(invoice_date), '+1 day') AS snapshot_date
    FROM transactions
),
rfm_raw AS (
    SELECT
        cs.customer_id,
        CAST(JULIANDAY((SELECT snapshot_date FROM snapshot))
             - JULIANDAY(cs.last_purchase) AS INTEGER) AS recency_days,
        cs.total_orders   AS frequency,
        cs.total_revenue  AS monetary
    FROM customer_summary cs
    WHERE cs.total_orders >= 1 AND cs.total_revenue > 0
),
rfm_ranked AS (
    SELECT *,
        RANK()       OVER (ORDER BY recency_days ASC)  AS recency_rank,
        DENSE_RANK() OVER (ORDER BY frequency DESC)    AS frequency_rank,
        RANK()       OVER (ORDER BY monetary DESC)     AS monetary_rank
    FROM rfm_raw
),
rfm_ntile AS (
    SELECT
        customer_id, recency_days, frequency, monetary,
        (6 - NTILE(5) OVER (ORDER BY recency_days ASC))    AS r_score,
        NTILE(5)      OVER (ORDER BY frequency_rank ASC)   AS f_score,
        NTILE(5)      OVER (ORDER BY monetary_rank ASC)    AS m_score
    FROM rfm_ranked
)
INSERT INTO rfm_scores (customer_id, recency_days, frequency, monetary, r_score, f_score, m_score)
SELECT customer_id, recency_days, frequency, monetary, r_score, f_score, m_score
FROM rfm_ntile
WHERE r_score BETWEEN 1 AND 5
  AND f_score BETWEEN 1 AND 5
  AND m_score BETWEEN 1 AND 5;


-- Populate rfm_segments with labels, churn, CLV
INSERT INTO rfm_segments (
    customer_id, recency_days, frequency, monetary,
    r_score, f_score, m_score, rfm_score,
    rfm_segment_code, segment_label, churn_probability, clv_estimate
)
SELECT
    rs.customer_id,
    rs.recency_days,
    rs.frequency,
    ROUND(rs.monetary, 2),
    rs.r_score,
    rs.f_score,
    rs.m_score,
    rs.rfm_score,
    rs.rfm_segment,
    CASE
        WHEN rs.rfm_score >= 12 THEN 'Champions'
        WHEN rs.rfm_score >= 9  THEN 'Loyal Customers'
        WHEN rs.rfm_score >= 7  THEN 'Potential Loyal'
        WHEN rs.rfm_score >= 5  THEN 'At Risk'
        ELSE                         'Lost / Inactive'
    END                                                     AS segment_label,
    CASE
        WHEN rs.recency_days > 180 THEN 0.85
        WHEN rs.recency_days > 90  THEN 0.60
        WHEN rs.recency_days > 30  THEN 0.25
        ELSE                            0.05
    END                                                     AS churn_probability,
    ROUND((rs.monetary / MAX(rs.recency_days, 1)) * 365, 2) AS clv_estimate
FROM rfm_scores rs;


-- ════════════════════════════════════════════════════════════
-- STEP 2: ANALYTICAL QUERIES ON RFM OUTPUT
-- These are the interview-level SQL queries recruiters expect.
-- Each query answers a real business question.
-- ════════════════════════════════════════════════════════════

-- Q1: How many customers in each segment?
SELECT
    segment_label,
    COUNT(*)                            AS customer_count,
    ROUND(COUNT(*) * 100.0
          / SUM(COUNT(*)) OVER (), 1)  AS pct_of_total
FROM rfm_segments
GROUP BY segment_label
ORDER BY customer_count DESC;


-- Q2: Average RFM metrics per segment
-- (The business profile of each segment)
SELECT
    segment_label,
    ROUND(AVG(recency_days), 1)         AS avg_recency_days,
    ROUND(AVG(frequency), 1)            AS avg_frequency,
    ROUND(AVG(monetary), 0)             AS avg_monetary_gbp,
    ROUND(AVG(rfm_score), 1)            AS avg_rfm_score,
    COUNT(*)                            AS customers
FROM rfm_segments
GROUP BY segment_label
ORDER BY avg_rfm_score DESC;


-- Q3: Revenue concentration (Pareto / 80-20 rule check)
-- Are 20% of customers generating 80% of revenue?
WITH ranked AS (
    SELECT
        customer_id,
        monetary,
        SUM(monetary) OVER ()                   AS total_revenue,
        ROW_NUMBER() OVER (ORDER BY monetary DESC) AS rn,
        COUNT(*) OVER ()                        AS total_customers
    FROM rfm_segments
),
cumulative AS (
    SELECT
        rn,
        monetary,
        total_revenue,
        total_customers,
        SUM(monetary) OVER (ORDER BY rn)        AS cumulative_revenue,
        ROUND(rn * 100.0 / total_customers, 0)  AS pct_customers
    FROM ranked
)
SELECT
    pct_customers                               AS top_n_pct_customers,
    ROUND(cumulative_revenue / total_revenue * 100, 1) AS pct_revenue_captured
FROM cumulative
WHERE pct_customers IN (10, 20, 30, 50)
ORDER BY pct_customers;


-- Q4: Churn risk summary
-- How much revenue is at risk of churning?
SELECT
    segment_label,
    CASE
        WHEN churn_probability >= 0.80 THEN 'Critical'
        WHEN churn_probability >= 0.55 THEN 'High'
        WHEN churn_probability >= 0.20 THEN 'Medium'
        ELSE 'Low'
    END                                         AS churn_risk_band,
    COUNT(*)                                    AS customers,
    ROUND(SUM(monetary), 0)                     AS revenue_at_risk
FROM rfm_segments
GROUP BY segment_label, churn_risk_band
ORDER BY revenue_at_risk DESC;


-- Q5: Top 10 highest-value customers (Champions)
SELECT
    seg.customer_id,
    c.country,
    seg.recency_days,
    seg.frequency,
    seg.monetary,
    seg.rfm_score,
    seg.clv_estimate,
    seg.segment_label
FROM rfm_segments seg
JOIN customers     c   ON c.customer_id = seg.customer_id
WHERE seg.segment_label = 'Champions'
ORDER BY seg.monetary DESC
LIMIT 10;


-- Q6: Month-over-month revenue trend
-- (From transactions table — shows SQL join + date aggregation)
SELECT
    SUBSTR(t.invoice_date, 1, 7)            AS year_month,
    COUNT(DISTINCT t.invoice)               AS orders,
    COUNT(DISTINCT t.customer_id)           AS active_customers,
    ROUND(SUM(t.quantity * t.unit_price), 0) AS revenue,
    ROUND(
        (SUM(t.quantity * t.unit_price)
         - LAG(SUM(t.quantity * t.unit_price))
           OVER (ORDER BY SUBSTR(t.invoice_date, 1, 7)))
        / NULLIF(LAG(SUM(t.quantity * t.unit_price))
           OVER (ORDER BY SUBSTR(t.invoice_date, 1, 7)), 0)
        * 100, 1)                           AS mom_growth_pct
FROM transactions t
GROUP BY year_month
ORDER BY year_month;


-- Q7: Customer cohort — first purchase month vs current segment
-- Shows how customers acquired in each cohort ended up segmenting
SELECT
    SUBSTR(cs.first_purchase, 1, 7)         AS acquisition_cohort,
    seg.segment_label,
    COUNT(*)                                AS customers,
    ROUND(AVG(seg.monetary), 0)             AS avg_spend
FROM rfm_segments     seg
JOIN customer_summary cs ON cs.customer_id = seg.customer_id
GROUP BY acquisition_cohort, segment_label
ORDER BY acquisition_cohort, customers DESC;


-- Q8: RFM score heatmap data (R vs F, aggregated by Monetary)
-- Useful for creating the heatmap in Power BI / Tableau
SELECT
    r_score,
    f_score,
    COUNT(*)                            AS customers,
    ROUND(AVG(monetary), 0)             AS avg_monetary,
    ROUND(AVG(m_score), 1)              AS avg_m_score
FROM rfm_segments
GROUP BY r_score, f_score
ORDER BY r_score DESC, f_score DESC;


-- Q9: Win-back candidate list for At Risk segment
-- Customers to target immediately (high historical spend, long inactive)
SELECT
    seg.customer_id,
    c.country,
    seg.recency_days,
    seg.frequency,
    seg.monetary,
    seg.clv_estimate,
    seg.churn_probability,
    -- Priority score: higher monetary + higher churn = more urgent
    ROUND(seg.monetary * seg.churn_probability, 0) AS win_back_priority_score
FROM rfm_segments seg
JOIN customers     c ON c.customer_id = seg.customer_id
WHERE seg.segment_label = 'At Risk'
ORDER BY win_back_priority_score DESC
LIMIT 50;


-- Q10: Product affinity by segment
-- Which products do Champions buy most? (multi-table join)
SELECT
    seg.segment_label,
    p.description,
    COUNT(*)                            AS purchase_count,
    ROUND(SUM(t.quantity * t.unit_price), 0) AS revenue
FROM rfm_segments  seg
JOIN transactions  t ON t.customer_id = seg.customer_id
JOIN products      p ON p.stock_code  = t.stock_code
WHERE seg.segment_label IN ('Champions', 'Loyal Customers')
GROUP BY seg.segment_label, p.description
ORDER BY seg.segment_label, revenue DESC
LIMIT 20;
