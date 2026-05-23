# ============================================================
# NOTEBOOK 03 — ML Clustering: K-Means + DBSCAN + PCA
# Project : Customer Segmentation using RFM + Clustering
# ============================================================
#
# WHAT THIS NOTEBOOK DOES (for recruiter reference):
# ─────────────────────────────────────────────────
# 1. Determines optimal number of clusters using Elbow + Silhouette methods
# 2. Fits K-Means clustering and assigns each customer a cluster
# 3. Validates with DBSCAN (density-based, handles noise/outliers)
# 4. Reduces to 2D using PCA for visualization
# 5. Profiles each cluster (mean RFM values, business interpretation)
# 6. Exports final segmented customer table for dashboard & reports
#
# KEY SKILLS DEMONSTRATED:
# ─────────────────────────────────────────────────
# ✅ Unsupervised ML (K-Means, DBSCAN)
# ✅ Dimensionality reduction (PCA)
# ✅ Model selection (Elbow method, Silhouette score)
# ✅ Statistical validation of clustering quality
# ✅ Business interpretation of ML outputs
# ✅ Interactive visualization with Plotly
# ============================================================

# %% — Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os

warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({
    'figure.facecolor':'white','axes.facecolor':'#f8f9fa',
    'axes.grid':True,'grid.alpha':0.4,
    'axes.spines.top':False,'axes.spines.right':False,
    'font.family':'DejaVu Sans','axes.titlesize':13,'axes.labelsize':11,
})

COLORS   = ['#2563EB','#10B981','#F59E0B','#EF4444','#8B5CF6','#06B6D4']
PROC_DIR = './data/processed/'

# %% — Load data
rfm        = pd.read_csv(f'{PROC_DIR}rfm_scored.csv')
rfm_scaled = np.load(f'{PROC_DIR}rfm_scaled_array.npy')

print(f"✅ RFM data loaded: {rfm.shape}")
print(f"   Scaled array   : {rfm_scaled.shape}")

# %% [markdown]
# ## 1. Determine Optimal Number of Clusters
#
# We use TWO complementary methods:
#
# **Elbow Method**: Plot Within-Cluster Sum of Squares (WCSS/inertia)
# vs K. The "elbow" (point of diminishing returns) is the optimal K.
# Limitation: subjective — the elbow isn't always obvious.
#
# **Silhouette Score**: Measures how similar a point is to its own
# cluster vs other clusters. Range: −1 to +1. Higher = better.
# More objective than the elbow method.
#
# **Best practice**: Use both together and pick the K that satisfies both.

# %% — Elbow + Silhouette analysis
K_range    = range(2, 12)
inertias   = []
silhouettes = []
db_scores  = []
ch_scores  = []

print("Computing clustering metrics for K = 2 to 11 …")
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=15, max_iter=300)
    labels = km.fit_predict(rfm_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(rfm_scaled, labels))
    db_scores.append(davies_bouldin_score(rfm_scaled, labels))
    ch_scores.append(calinski_harabasz_score(rfm_scaled, labels))
    print(f"  K={k}  Silhouette={silhouettes[-1]:.3f}  DB={db_scores[-1]:.3f}")

# %% — Plot Elbow + Silhouette
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Optimal K Selection', fontweight='bold', fontsize=14)

k_list = list(K_range)

# Elbow
axes[0].plot(k_list, inertias, 'o-', color=COLORS[0], linewidth=2.5, markersize=7)
axes[0].set_title('Elbow Method (WCSS / Inertia)', fontweight='bold')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')

# Silhouette
axes[1].plot(k_list, silhouettes, 's-', color=COLORS[1], linewidth=2.5, markersize=7)
best_k_idx = np.argmax(silhouettes)
best_k     = k_list[best_k_idx]
axes[1].axvline(best_k, color=COLORS[3], linestyle='--', linewidth=1.5,
                label=f'Best K = {best_k}')
axes[1].scatter([best_k], [silhouettes[best_k_idx]], color=COLORS[3], s=120, zorder=5)
axes[1].set_title('Silhouette Score (higher = better)', fontweight='bold')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].legend()

plt.tight_layout()
plt.savefig('./reports/11_optimal_k.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n✅ Optimal K based on Silhouette: K = {best_k}")

# %% [markdown]
# ## 2. Fit Final K-Means Model
#
# We use K=5 (or the optimal K found above) because it produces
# 5 actionable business segments: Champions, Loyal, Potential Loyal, At Risk, Lost.
#
# **n_init=20**: Run K-Means 20 times with different centroid seeds
# and keep the best result (lowest inertia). This avoids bad local optima.

# %% — Final K-Means
OPTIMAL_K = max(5, best_k)  # at least 5 for business meaning
print(f"🎯 Fitting final K-Means with K = {OPTIMAL_K} …")

km_final = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=20, max_iter=500)
rfm['Cluster'] = km_final.fit_predict(rfm_scaled)

# Cluster validation scores
sil  = silhouette_score(rfm_scaled, rfm['Cluster'])
db   = davies_bouldin_score(rfm_scaled, rfm['Cluster'])
ch   = calinski_harabasz_score(rfm_scaled, rfm['Cluster'])

print(f"\n--- K-Means Model Quality ---")
print(f"  Silhouette Score          : {sil:.4f}  (>0.5 = good, >0.7 = excellent)")
print(f"  Davies-Bouldin Score      : {db:.4f}  (lower = better)")
print(f"  Calinski-Harabasz Score   : {ch:.1f}  (higher = better)")

# %% — Assign business labels based on cluster profiles
cluster_profile = rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean()
print("\n--- Raw Cluster Profiles ---")
print(cluster_profile.round(1).to_string())

# Auto-label: Champions = lowest Recency + highest Frequency + highest Monetary
recency_rank   = cluster_profile['Recency'].rank()
frequency_rank = cluster_profile['Frequency'].rank(ascending=False)
monetary_rank  = cluster_profile['Monetary'].rank(ascending=False)
composite_rank = recency_rank + frequency_rank + monetary_rank

sorted_clusters = composite_rank.sort_values().index.tolist()

# Auto-label business segments
if OPTIMAL_K >= 5:
    label_map = {
        sorted_clusters[0]: 'Champions',
        sorted_clusters[1]: 'Loyal Customers',
        sorted_clusters[2]: 'Potential Loyal',
        sorted_clusters[3]: 'At Risk',
        sorted_clusters[4]: 'Lost / Inactive'
    }

    for i in range(5, OPTIMAL_K):
        label_map[sorted_clusters[i]] = f'Segment {i+1}'

else:
    labels_list = [
        'Champions',
        'Loyal Customers',
        'Potential Loyal',
        'At Risk',
        'Lost / Inactive'
    ]

    label_map = {
        c: labels_list[i]
        for i, c in enumerate(sorted_clusters)
    }
rfm['Segment'] = rfm['Cluster'].map(label_map)

print("\n--- Labeled Cluster → Segment Mapping ---")
for cluster, segment in sorted(label_map.items()):
    count = (rfm['Cluster'] == cluster).sum()
    print(f"  Cluster {cluster} → {segment:<20} ({count:,} customers)")

# %% [markdown]
# ## 3. DBSCAN — Alternative Clustering
#
# **Why DBSCAN?**: K-Means assumes spherical clusters and requires K upfront.
# DBSCAN (Density-Based Spatial Clustering) finds clusters of arbitrary shape
# and automatically identifies OUTLIERS (label = −1).
#
# **Use case in business**: DBSCAN outliers = very unusual customers
# (e.g., one-time bulk buyers) who shouldn't be in any regular segment.
#
# **Parameters**:
# - eps     : neighborhood radius (tuned using k-distance graph)
# - min_samples: minimum points to form a dense region

# %% — DBSCAN
from sklearn.neighbors import NearestNeighbors

# Find optimal eps using k-distance graph (k = min_samples)
min_samples = 5
nbrs = NearestNeighbors(n_neighbors=min_samples).fit(rfm_scaled)
distances, _ = nbrs.kneighbors(rfm_scaled)
distances     = np.sort(distances[:, -1])

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(distances, color=COLORS[0], linewidth=1.5)
ax.set_title('K-Distance Graph (for DBSCAN eps selection)', fontweight='bold')
ax.set_xlabel('Points sorted by distance')
ax.set_ylabel(f'{min_samples}-NN Distance')
plt.tight_layout()
plt.savefig('./reports/12_dbscan_kdistance.png', dpi=150, bbox_inches='tight')
plt.show()

# Fit DBSCAN
eps_val = float(np.percentile(distances, 85))
dbscan  = DBSCAN(eps=eps_val, min_samples=min_samples)
rfm['DBSCAN_Cluster'] = dbscan.fit_predict(rfm_scaled)

n_clusters_db = len(set(rfm['DBSCAN_Cluster'])) - (1 if -1 in rfm['DBSCAN_Cluster'].values else 0)
n_noise       = (rfm['DBSCAN_Cluster'] == -1).sum()

print(f"\n--- DBSCAN Results ---")
print(f"  eps         : {eps_val:.3f}")
print(f"  min_samples : {min_samples}")
print(f"  Clusters    : {n_clusters_db}")
print(f"  Noise points: {n_noise:,} ({n_noise/len(rfm)*100:.1f}%)")
print("  Note: Noise points = unusual customers that don't fit any cluster")
print("        → worth investigating as VIP outliers or fraud signals")

# %% [markdown]
# ## 4. PCA — 2D Visualization
#
# **Why PCA?**: Our RFM space is 3-dimensional. To visualize clusters,
# we reduce to 2 principal components that capture maximum variance.
#
# **Explained variance**: We report how much information is retained.
# >80% explained variance means the 2D plot is a faithful representation.

# %% — PCA reduction
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(rfm_scaled)

rfm['PC1'] = pca_coords[:, 0]
rfm['PC2'] = pca_coords[:, 1]

ev = pca.explained_variance_ratio_
print(f"✅ PCA complete")
print(f"   PC1 explains : {ev[0]*100:.1f}% variance")
print(f"   PC2 explains : {ev[1]*100:.1f}% variance")
print(f"   Total        : {sum(ev)*100:.1f}% variance retained in 2D")

# %% — Static PCA scatter (matplotlib)
fig, ax = plt.subplots(figsize=(12, 8))
segment_order = ['Champions','Loyal Customers','At Risk','Lost / Inactive']
color_map     = dict(zip(segment_order, COLORS))

for seg in rfm['Segment'].unique():
    mask = rfm['Segment'] == seg
    ax.scatter(rfm.loc[mask, 'PC1'], rfm.loc[mask, 'PC2'],
               alpha=0.5, s=18, label=seg,
               color=color_map.get(seg, '#9CA3AF'))

ax.set_title(f'Customer Segments — PCA 2D View\n(explains {sum(ev)*100:.0f}% of variance)',
             fontweight='bold')
ax.set_xlabel(f'Principal Component 1 ({ev[0]*100:.1f}%)')
ax.set_ylabel(f'Principal Component 2 ({ev[1]*100:.1f}%)')
ax.legend(markerscale=2, fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig('./reports/13_pca_clusters.png', dpi=150, bbox_inches='tight')
plt.show()

# %% — Interactive Plotly scatter
fig_px = px.scatter(
    rfm, x='PC1', y='PC2', color='Segment',
    hover_data={'Recency':True, 'Frequency':True,
                'Monetary':':.2f', 'RFM_Score':True,
                'PC1':False, 'PC2':False},
    color_discrete_map=color_map,
    title=f'Interactive Customer Segments (PCA 2D — {sum(ev)*100:.0f}% variance)',
    template='plotly_white',
    opacity=0.7
)
fig_px.update_traces(marker=dict(size=5))
fig_px.update_layout(legend=dict(orientation='h', y=-0.15))
fig_px.write_html('./reports/13_pca_interactive.html')
print("💾 Interactive chart saved: reports/13_pca_interactive.html")
fig_px.show()

# %% [markdown]
# ## 5. Cluster Profiling
#
# This is the section that bridges ML → Business.
# Each cluster gets a detailed profile that marketing can act on.

# %% — Detailed cluster profiles
print("\n" + "="*65)
print("CLUSTER PROFILES — BUSINESS SUMMARY")
print("="*65)

profile_cols = ['Recency','Frequency','Monetary','RFM_Score','R_Score','F_Score','M_Score']
profiles = rfm.groupby('Segment')[profile_cols].agg(['mean','median']).round(1)

summary = rfm.groupby('Segment').agg(
    Customers       = ('Customer ID', 'count'),
    Avg_Recency     = ('Recency',     'mean'),
    Avg_Frequency   = ('Frequency',   'mean'),
    Avg_Monetary    = ('Monetary',    'mean'),
    Total_Revenue   = ('Monetary',    'sum'),
    Avg_RFM_Score   = ('RFM_Score',   'mean'),
).round(1)
summary['Pct_Customers'] = (summary['Customers'] / len(rfm) * 100).round(1)
summary['Pct_Revenue']   = (summary['Total_Revenue'] / rfm['Monetary'].sum() * 100).round(1)

print(summary.to_string())

# %% — CLV estimate
# Simple CLV proxy: projected annual spend based on purchase rate
# CLV = (Avg Monthly Spend × 12) × (1 / Churn Probability)
# Simplified: CLV ≈ (Monetary / Recency_days) × 365
rfm['CLV_Estimate'] = (rfm['Monetary'] / rfm['Recency'].clip(lower=1)) * 365
rfm['CLV_Estimate'] = rfm['CLV_Estimate'].round(2)

clv_by_segment = rfm.groupby('Segment')['CLV_Estimate'].mean().sort_values(ascending=False)
print("\n--- Estimated Annual CLV by Segment ---")
for seg, clv in clv_by_segment.items():
    print(f"  {seg:<20}: £{clv:,.0f}")

# %% — Save final output
rfm.to_csv(f'{PROC_DIR}rfm_final.csv', index=False)
summary.reset_index().to_csv(f'{PROC_DIR}segment_summary.csv', index=False)

print(f"\n✅ Final outputs saved:")
print(f"   {PROC_DIR}rfm_final.csv       ← full table: RFM + Cluster + Segment + CLV")
print(f"   {PROC_DIR}segment_summary.csv ← aggregate summary for Power BI / Tableau")
print("\n📌 Next: Run 04_insights.py for business report generation.")
print("         Run dashboard/app.py to launch Streamlit dashboard.")
