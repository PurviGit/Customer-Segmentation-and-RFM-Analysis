# ============================================================
# FILE     : anomaly_detection/anomaly_detector.py
# PROJECT  : Customer Segmentation — Anomaly Detection
# ============================================================
#
# WHAT IS ISOLATION FOREST? (explain this in interviews)
# ───────────────────────────────────────────────────────
# Isolation Forest is an UNSUPERVISED anomaly detection algorithm.
# It works on a beautifully simple idea:
#
#   Anomalies are RARE and DIFFERENT from normal data.
#   Therefore, they are EASIER TO ISOLATE (separate) than
#   normal points using random splits.
#
# Think of it like a game of 20 questions on a feature space:
#   - Pick a random feature and a random split value
#   - Repeat until you've isolated every point
#   - Anomalies get isolated in FEWER steps
#     (fewer questions needed to separate them)
#   - Normal points need MANY steps (they're surrounded by similar data)
#
# The "anomaly score" = average path length across many trees.
# Short path = anomaly. Long path = normal.
#
# WHY USE IT HERE?
# ───────────────────────────────────────────────────────
# In retail transaction data, anomalies can mean:
#   1. WHOLESALE/BULK ORDERS: a business buying 500 items at once
#      → Should be EXCLUDED from RFM segmentation (not retail customers)
#   2. FRAUD: unusual spend patterns or velocity
#   3. DATA ENTRY ERRORS: £10,000 single transaction
#   4. CORPORATE ACCOUNTS: different buying behaviour from retail
#
# Flagging these BEFORE segmentation makes clusters cleaner.
#
# RECRUITER ONE-LINER:
# "I used Isolation Forest — an unsupervised tree-based algorithm —
#  to detect anomalous transactions (bulk buyers, potential fraud,
#  data errors) before running K-Means. This prevented outliers
#  from distorting cluster centroids, producing cleaner segments."
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#f8f9fa',
    'axes.grid': True, 'grid.alpha': 0.4,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'DejaVu Sans', 'axes.titlesize': 13,
})

COLORS   = ['#2563EB', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6']
PROC_DIR = Path('./data/processed')
REP_DIR  = Path('./reports')
REP_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  ANOMALY DETECTION — Isolation Forest")
print("=" * 60)


# ════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# Two-level anomaly detection:
#   Level 1: Transaction-level (flag individual suspicious orders)
#   Level 2: Customer-level (flag unusual customers in RFM space)
# ════════════════════════════════════════════════════════════

def load_data():
    """Load transaction and RFM data."""
    # Transaction data
    tx_path  = PROC_DIR / 'retail_cleaned.csv'
    rfm_path = PROC_DIR / 'rfm_scored.csv'

    if tx_path.exists():
        df_tx = pd.read_csv(tx_path)
    else:
        print("⚠️  Generating synthetic transaction data for anomaly detection")
        np.random.seed(42)
        n = 20000
        df_tx = pd.DataFrame({
            'Customer ID'  : np.random.choice(range(10000, 12000), n),
            'Invoice'      : [f'{500000+i}' for i in range(n)],
            'Quantity'     : np.random.choice(
                                list(range(1, 12)) + [50, 100, 200, 500, 1000],
                                n, p=[0.08]*11 + [0.02, 0.01, 0.01, 0.005, 0.005]),
            'TotalAmount'  : np.round(np.random.lognormal(3.8, 1.1, n), 2),
            'Country'      : np.random.choice(
                                ['United Kingdom','Germany','France'], n, p=[0.85,0.08,0.07]),
        })
        # Inject obvious anomalies
        anomaly_idx = np.random.choice(n, 150, replace=False)
        df_tx.loc[anomaly_idx[:50], 'Quantity']    = np.random.randint(500, 2000, 50)
        df_tx.loc[anomaly_idx[:50], 'TotalAmount'] = df_tx.loc[anomaly_idx[:50], 'TotalAmount'] * 20
        df_tx.loc[anomaly_idx[50:100], 'TotalAmount'] = np.random.uniform(5000, 50000, 50)

    if rfm_path.exists():
        df_rfm = pd.read_csv(rfm_path)
    else:
        print("⚠️  Generating synthetic RFM data")
        n_cust = 1500
        df_rfm = pd.DataFrame({
            'Customer ID': range(10000, 10000 + n_cust),
            'Recency'    : np.random.exponential(60, n_cust).clip(1, 365).astype(int),
            'Frequency'  : np.random.negative_binomial(2, 0.4, n_cust).clip(1, 100),
            'Monetary'   : np.round(np.random.lognormal(5.5, 1.2, n_cust), 2),
            'Segment'    : np.random.choice(
                              ['Champions','Loyal Customers','Potential Loyal',
                               'At Risk','Lost / Inactive'], n_cust),
        })
        # Inject customer-level anomalies
        df_rfm.loc[:30, 'Monetary']  = np.random.uniform(20000, 100000, 31)
        df_rfm.loc[:30, 'Frequency'] = np.random.randint(50, 200, 31)

    return df_tx, df_rfm

df_tx, df_rfm = load_data()
print(f"✅ Transaction data: {len(df_tx):,} rows")
print(f"   RFM data        : {len(df_rfm):,} customers")


# ════════════════════════════════════════════════════════════
# STEP 2: TRANSACTION-LEVEL ANOMALY DETECTION
#
# Features used to detect anomalous transactions:
#   - Quantity        : bulk orders are anomalous
#   - TotalAmount     : very high revenue per order is anomalous
#   - Implied price   : TotalAmount / Quantity (unit price proxy)
#
# contamination: the proportion of data we EXPECT to be anomalous
# We set it to 2% — conservative, avoids too many false positives
# ════════════════════════════════════════════════════════════

print("\n[Step 2] Transaction-level anomaly detection …")

# Feature engineering for transactions
df_tx['implied_unit_price'] = (
    df_tx['TotalAmount'] / df_tx['Quantity'].clip(lower=1)
).clip(upper=500)

tx_features = ['Quantity', 'TotalAmount', 'implied_unit_price']
tx_features  = [f for f in tx_features if f in df_tx.columns]

# Scale features (Isolation Forest is not distance-based but scaling helps)
scaler_tx = StandardScaler()
X_tx = scaler_tx.fit_transform(df_tx[tx_features])

# Fit Isolation Forest
# n_estimators=200: number of trees (more = more stable results)
# contamination=0.02: we expect ~2% of transactions to be anomalous
# max_samples='auto': uses min(256, n_samples) — fast and accurate
iso_tx = IsolationForest(
    n_estimators  = 200,
    contamination = 0.02,
    max_samples   = 'auto',
    random_state  = 42,
    n_jobs        = -1,
)
df_tx['anomaly_score']    = iso_tx.fit_predict(X_tx)  # -1 = anomaly, 1 = normal
df_tx['anomaly_score_raw']= iso_tx.score_samples(X_tx) # raw score (lower = more anomalous)
df_tx['is_anomaly']       = df_tx['anomaly_score'] == -1

n_anomalies_tx = df_tx['is_anomaly'].sum()
print(f"   Transactions flagged as anomalous: {n_anomalies_tx:,} ({n_anomalies_tx/len(df_tx)*100:.1f}%)")

print("\n--- Anomalous Transactions Sample ---")
anomaly_sample = df_tx[df_tx['is_anomaly']].nlargest(10, 'TotalAmount')
print(anomaly_sample[tx_features + ['is_anomaly', 'anomaly_score_raw']].to_string())

# Anomaly type classification
def classify_tx_anomaly(row):
    if row['Quantity'] > 200:
        return 'Bulk Order'
    elif row['TotalAmount'] > df_tx['TotalAmount'].quantile(0.998):
        return 'Extreme Value'
    elif row['implied_unit_price'] > df_tx['implied_unit_price'].quantile(0.998):
        return 'Unusual Price'
    else:
        return 'Statistical Outlier'

df_tx.loc[df_tx['is_anomaly'], 'anomaly_type'] = df_tx[df_tx['is_anomaly']].apply(
    classify_tx_anomaly, axis=1)

print("\n--- Anomaly Type Breakdown ---")
if df_tx['is_anomaly'].any():
    print(df_tx[df_tx['is_anomaly']]['anomaly_type'].value_counts().to_string())


# ════════════════════════════════════════════════════════════
# STEP 3: CUSTOMER-LEVEL ANOMALY DETECTION
#
# Run Isolation Forest on the RFM space.
# Detects customers who are outliers in their purchase behaviour:
#   - Very high frequency + very high monetary (wholesale buyers)
#   - Very low recency + zero frequency (single burst purchaser)
#   - Unusual combinations of R, F, M scores
# ════════════════════════════════════════════════════════════

print("\n[Step 3] Customer-level anomaly detection …")

rfm_features = ['Recency', 'Frequency', 'Monetary']
rfm_features  = [f for f in rfm_features if f in df_rfm.columns]

# Log transform to reduce skewness (same as ML pipeline)
X_rfm_raw = df_rfm[rfm_features].copy()
X_rfm_log = np.log1p(X_rfm_raw)

scaler_rfm = StandardScaler()
X_rfm = scaler_rfm.fit_transform(X_rfm_log)

iso_cust = IsolationForest(
    n_estimators  = 300,
    contamination = 0.03,   # expect ~3% unusual customers
    max_samples   = 'auto',
    random_state  = 42,
    n_jobs        = -1,
)
df_rfm['anomaly_score']    = iso_cust.fit_predict(X_rfm)
df_rfm['anomaly_score_raw']= iso_cust.score_samples(X_rfm)
df_rfm['is_anomaly']       = df_rfm['anomaly_score'] == -1

n_anomalies_cust = df_rfm['is_anomaly'].sum()
print(f"   Customers flagged as anomalous: {n_anomalies_cust:,} ({n_anomalies_cust/len(df_rfm)*100:.1f}%)")

# Anomaly customer profile
print("\n--- Normal vs Anomaly Customer Profiles ---")
comparison = df_rfm.groupby('is_anomaly')[rfm_features].mean().round(1)
comparison.index = ['Normal', 'Anomaly']
print(comparison.to_string())

# Classify customer anomalies
def classify_cust_anomaly(row):
    if row['Monetary'] > df_rfm['Monetary'].quantile(0.99):
        return 'Possible Wholesale'
    elif row['Frequency'] > df_rfm['Frequency'].quantile(0.99):
        return 'Power Buyer'
    elif row['Recency'] < 7 and row['Frequency'] > 20:
        return 'Suspicious Velocity'
    else:
        return 'Statistical Outlier'

df_rfm.loc[df_rfm['is_anomaly'], 'anomaly_type'] = df_rfm[df_rfm['is_anomaly']].apply(
    classify_cust_anomaly, axis=1)

print("\n--- Customer Anomaly Types ---")
if df_rfm['is_anomaly'].any():
    print(df_rfm[df_rfm['is_anomaly']]['anomaly_type'].value_counts().to_string())


# ════════════════════════════════════════════════════════════
# STEP 4: VISUALISATIONS
# ════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Anomaly Detection — Isolation Forest', fontweight='bold', fontsize=15)

# Plot 1: Transaction anomalies — Quantity vs TotalAmount
ax = axes[0, 0]
normal_tx  = df_tx[~df_tx['is_anomaly']]
anomaly_tx = df_tx[ df_tx['is_anomaly']]
ax.scatter(normal_tx['Quantity'].clip(upper=300), normal_tx['TotalAmount'].clip(upper=2000),
           alpha=0.3, s=8, color=COLORS[0], label=f'Normal ({len(normal_tx):,})')
ax.scatter(anomaly_tx['Quantity'].clip(upper=300), anomaly_tx['TotalAmount'].clip(upper=2000),
           alpha=0.8, s=30, color=COLORS[1], label=f'Anomaly ({len(anomaly_tx):,})', marker='x')
ax.set_title('Transaction Anomalies\nQuantity vs Total Amount', fontweight='bold')
ax.set_xlabel('Quantity (capped at 300)')
ax.set_ylabel('Total Amount £ (capped at £2,000)')
ax.legend(fontsize=9)

# Plot 2: Anomaly score distribution (transactions)
ax = axes[0, 1]
ax.hist(df_tx[~df_tx['is_anomaly']]['anomaly_score_raw'], bins=50,
        color=COLORS[0], alpha=0.7, label='Normal', density=True)
ax.hist(df_tx[ df_tx['is_anomaly']]['anomaly_score_raw'], bins=20,
        color=COLORS[1], alpha=0.7, label='Anomaly', density=True)
ax.set_title('Anomaly Score Distribution\n(Transactions)', fontweight='bold')
ax.set_xlabel('Isolation Forest Score (lower = more anomalous)')
ax.set_ylabel('Density')
ax.legend(fontsize=9)

# Plot 3: Customer anomaly type breakdown
ax = axes[0, 2]
if df_rfm['is_anomaly'].any() and 'anomaly_type' in df_rfm.columns:
    types = df_rfm[df_rfm['is_anomaly']]['anomaly_type'].value_counts()
    ax.barh(types.index, types.values, color=COLORS[1], alpha=0.85, edgecolor='white')
    for i, (idx, val) in enumerate(types.items()):
        ax.text(val + 0.2, i, str(val), va='center', fontsize=10, fontweight='bold')
    ax.set_title('Customer Anomaly Types', fontweight='bold')
    ax.set_xlabel('Count')

# Plot 4: PCA — normal vs anomaly customers
ax = axes[1, 0]
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_rfm)
ev = pca.explained_variance_ratio_

normal_mask  = ~df_rfm['is_anomaly'].values
anomaly_mask =  df_rfm['is_anomaly'].values

ax.scatter(X_pca[normal_mask,  0], X_pca[normal_mask,  1],
           alpha=0.3, s=10, color=COLORS[0], label=f'Normal ({normal_mask.sum():,})')
ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1],
           alpha=0.9, s=50, color=COLORS[1], label=f'Anomaly ({anomaly_mask.sum():,})',
           marker='*', zorder=5)
ax.set_title(f'PCA View: Normal vs Anomaly\n({sum(ev)*100:.0f}% variance)', fontweight='bold')
ax.set_xlabel(f'PC1 ({ev[0]*100:.0f}%)')
ax.set_ylabel(f'PC2 ({ev[1]*100:.0f}%)')
ax.legend(fontsize=9)

# Plot 5: RFM comparison — normal vs anomaly boxplot
ax = axes[1, 1]
data_normal  = df_rfm[~df_rfm['is_anomaly']]['Monetary'].clip(upper=df_rfm['Monetary'].quantile(0.99))
data_anomaly = df_rfm[ df_rfm['is_anomaly']]['Monetary'].clip(upper=df_rfm['Monetary'].quantile(0.99))
bp = ax.boxplot([data_normal, data_anomaly], patch_artist=True,
                labels=['Normal', 'Anomaly'])
bp['boxes'][0].set_facecolor(COLORS[0] + '80')
bp['boxes'][1].set_facecolor(COLORS[1] + '80')
ax.set_title('Monetary Distribution\nNormal vs Anomaly Customers', fontweight='bold')
ax.set_ylabel('Monetary Value (£)')

# Plot 6: Impact on segmentation — what % of each segment is anomalous
ax = axes[1, 2]
if 'Segment' in df_rfm.columns and df_rfm['Segment'].notna().any():
    seg_anomaly = df_rfm.groupby('Segment').agg(
        total   = ('is_anomaly', 'count'),
        anomaly = ('is_anomaly', 'sum'),
    )
    seg_anomaly['pct'] = (seg_anomaly['anomaly'] / seg_anomaly['total'] * 100).round(1)
    colors_seg = [COLORS[i % len(COLORS)] for i in range(len(seg_anomaly))]
    bars = ax.bar(seg_anomaly.index, seg_anomaly['pct'], color=colors_seg, alpha=0.85, edgecolor='white')
    for bar, pct in zip(bars, seg_anomaly['pct']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{pct:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Anomaly Rate by Segment', fontweight='bold')
    ax.set_ylabel('% Flagged as Anomaly')
    ax.set_xticklabels(seg_anomaly.index, rotation=20, ha='right', fontsize=8)

plt.tight_layout()
plt.savefig(REP_DIR / 'anomaly_detection_results.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n📊 Chart saved: reports/anomaly_detection_results.png")


# ════════════════════════════════════════════════════════════
# STEP 5: BUSINESS RECOMMENDATIONS
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  BUSINESS RECOMMENDATIONS")
print("=" * 60)

revenue_in_anomalies = 0
if 'TotalAmount' in df_tx.columns:
    revenue_in_anomalies = df_tx[df_tx['is_anomaly']]['TotalAmount'].sum()
    total_revenue        = df_tx['TotalAmount'].sum()
    print(f"""
  Transaction anomalies:
    Flagged: {n_anomalies_tx:,} transactions ({n_anomalies_tx/len(df_tx)*100:.1f}%)
    Revenue in anomalies: £{revenue_in_anomalies:,.0f} ({revenue_in_anomalies/total_revenue*100:.1f}% of total)
    Action: Review manually — may be B2B / wholesale accounts

  Customer anomalies:
    Flagged: {n_anomalies_cust:,} customers ({n_anomalies_cust/len(df_rfm)*100:.1f}%)
    Action:
      - Wholesale/Bulk buyers → move to separate B2B segment
      - Suspicious velocity   → flag for fraud review
      - Statistical outliers  → exclude from RFM clustering
                                to prevent centroid distortion
""")

# Save outputs
df_tx[df_tx['is_anomaly']].to_csv(PROC_DIR / 'anomalous_transactions.csv', index=False)
df_rfm[df_rfm['is_anomaly']].to_csv(PROC_DIR / 'anomalous_customers.csv',  index=False)
df_rfm.to_csv(PROC_DIR / 'rfm_with_anomaly_flags.csv', index=False)

print(f"✅ Outputs saved:")
print(f"   anomalous_transactions.csv — {n_anomalies_tx:,} flagged transactions")
print(f"   anomalous_customers.csv    — {n_anomalies_cust:,} flagged customers")
print(f"   rfm_with_anomaly_flags.csv — full RFM table with anomaly column")
print("\n📌 Next: python cohort_analysis/cohort_retention.py")
