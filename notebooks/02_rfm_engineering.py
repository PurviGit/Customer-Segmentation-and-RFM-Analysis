# ============================================================
# NOTEBOOK 02 (UPDATED) — RFM Engineering
# ============================================================
# NOTEBOOK 02 — RFM Feature Engineering
# Project : Customer Segmentation using RFM + Clustering
# ============================================================
#
# WHAT THIS NOTEBOOK DOES (for recruiter reference):
# ─────────────────────────────────────────────────
# 1. Computes Recency, Frequency, Monetary for every customer
# 2. Applies quintile-based RFM scoring (1–5 scale)
# 3. Creates composite RFM score and segment labels
# 4. Normalizes features for downstream ML clustering
# 5. Visualizes RFM distributions, correlations, segment maps
# 6. Saves RFM table as both CSV and the ML-ready scaled array
#
# RFM THEORY (document this for interviewers):
# ─────────────────────────────────────────────
# Recency   : How recently did the customer buy?
#             → Lower = better (bought recently = still engaged)
# Frequency : How often do they buy?
#             → Higher = better (loyal, habitual customer)
# Monetary  : How much total have they spent?
#             → Higher = better (high-value customer)
#
# RFM was developed in direct-mail marketing (1980s) and remains
# one of the most effective customer valuation frameworks in retail,
# e-commerce, and subscription businesses.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from pathlib import Path

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#f8f9fa',
    'axes.grid': True, 'grid.alpha': 0.4,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'DejaVu Sans', 'axes.titlesize': 13,
})

COLORS   = ['#2563EB','#10B981','#F59E0B','#EF4444','#8B5CF6']
PROC_DIR = Path('./data/processed')

# ── Smart data loading: SQL output preferred ─────────────────
SQL_OUTPUT = PROC_DIR / 'rfm_from_sql.csv'
CLEANED    = PROC_DIR / 'retail_cleaned.csv'

if SQL_OUTPUT.exists():
    print("📊 Loading RFM data from SQL pipeline output …")
    rfm = pd.read_csv(SQL_OUTPUT)
    # Rename to match expected columns if coming from SQL
    rename_map = {'recency': 'Recency', 'frequency': 'Frequency',
                  'monetary': 'Monetary', 'r_score': 'R_Score',
                  'f_score': 'F_Score', 'm_score': 'M_Score',
                  'rfm_score': 'RFM_Score', 'segment': 'Segment',
                  'customer_id': 'Customer ID', 'country': 'Country',
                  'churn_probability': 'Churn_Probability',
                  'clv_estimate': 'CLV_Estimate',
                  'rfm_segment_code': 'RFM_Segment'}
    rfm.rename(columns={k:v for k,v in rename_map.items() if k in rfm.columns}, inplace=True)
    SOURCE = "SQL"
    print(f"   ✅ {len(rfm):,} customers loaded from SQL output")
    print(f"   Columns: {list(rfm.columns)}")

elif CLEANED.exists():
    print("⚠️  SQL output not found — computing RFM from Python (run src/db_connector.py first for SQL path)")
    df = pd.read_csv(CLEANED, parse_dates=['InvoiceDate'])
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('Customer ID').agg(
        Recency   = ('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency = ('Invoice', 'nunique'),
        Monetary  = ('TotalAmount', 'sum')
    ).reset_index()
    rfm['Monetary'] = rfm['Monetary'].round(2)

    def safe_qcut(series, q=5, labels=None, ascending=True):
        ranked = series.rank(method='first', ascending=ascending)
        return pd.qcut(ranked, q=q, labels=labels)

    rfm['R_Score'] = safe_qcut(rfm['Recency'],   ascending=True,  labels=[5,4,3,2,1]).astype(int)
    rfm['F_Score'] = safe_qcut(rfm['Frequency'], ascending=False, labels=[5,4,3,2,1]).astype(int)
    rfm['M_Score'] = safe_qcut(rfm['Monetary'],  ascending=False, labels=[5,4,3,2,1]).astype(int)
    rfm['RFM_Score']   = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    rfm['Segment'] = rfm['RFM_Score'].apply(
        lambda s: 'Champions' if s >= 12 else
                  'Loyal Customers' if s >= 9 else
                  'Potential Loyal' if s >= 7 else
                  'At Risk' if s >= 5 else 'Lost / Inactive')
    rfm['CLV_Estimate'] = (rfm['Monetary'] / rfm['Recency'].clip(lower=1) * 365).round(2)
    SOURCE = "Python"
    print(f"   ✅ {len(rfm):,} customers computed via Python")

else:
    print("❌ No data found. Run: python src/db_connector.py")
    raise FileNotFoundError("No processed data available")

print(f"\n📌 Data source: {SOURCE}")
print(f"   Shape: {rfm.shape}")
print(f"\n--- RFM Summary ---")
print(rfm[['Recency','Frequency','Monetary']].describe().round(2))

# ── Log transformation + StandardScaler ──────────────────────
rfm['log_Recency']   = np.log1p(rfm['Recency'])
rfm['log_Frequency'] = np.log1p(rfm['Frequency'])
rfm['log_Monetary']  = np.log1p(rfm['Monetary'])

features = ['log_Recency', 'log_Frequency', 'log_Monetary']
scaler   = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[features])

rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['R_scaled','F_scaled','M_scaled'])
rfm_scaled_df['Customer ID'] = rfm['Customer ID'].values

# ── Visualizations (same as before) ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'RFM Distributions (Source: {SOURCE})', fontweight='bold')

for ax, (col, label, color) in zip(axes, [
    ('Recency','Days Since Last Purchase', COLORS[3]),
    ('Frequency','Number of Orders', COLORS[0]),
    ('Monetary','Total Spend (£)', COLORS[1]),
]):
    ax.hist(rfm[col], bins=40, color=color, alpha=0.8, edgecolor='white')
    ax.axvline(rfm[col].mean(),   color='black', linestyle='--', lw=1.5,
               label=f'Mean: {rfm[col].mean():.1f}')
    ax.axvline(rfm[col].median(), color='gray',  linestyle=':',  lw=1.5,
               label=f'Median: {rfm[col].median():.1f}')
    ax.set_title(label, fontweight='bold')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('./reports/07_rfm_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Segment distribution ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
seg_counts = rfm['Segment'].value_counts()
colors_seg = dict(zip(seg_counts.index, COLORS))
bars = ax.barh(seg_counts.index, seg_counts.values,
               color=[colors_seg.get(s, '#ccc') for s in seg_counts.index],
               edgecolor='white', alpha=0.9)
for bar, val in zip(bars, seg_counts.values):
    pct = val / len(rfm) * 100
    ax.text(val + 5, bar.get_y() + bar.get_height()/2,
            f'{val:,}  ({pct:.1f}%)', va='center', fontsize=10)
ax.set_title(f'Customer Segment Distribution (Source: {SOURCE})', fontweight='bold')
ax.set_xlabel('Number of Customers')
plt.tight_layout()
plt.savefig('./reports/08_segment_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Save outputs ──────────────────────────────────────────────
rfm.to_csv(PROC_DIR / 'rfm_scored.csv', index=False)
rfm_scaled_df.to_csv(PROC_DIR / 'rfm_scaled.csv', index=False)
np.save(PROC_DIR / 'rfm_scaled_array.npy', rfm_scaled)

print(f"\n✅ Outputs saved:")
print(f"   rfm_scored.csv       — {len(rfm):,} rows (source: {SOURCE})")
print(f"   rfm_scaled.csv       — scaled features")
print(f"   rfm_scaled_array.npy — numpy array for clustering")
print("\n📌 Next: python notebooks/03_clustering.py")
