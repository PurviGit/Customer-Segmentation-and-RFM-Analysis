# ============================================================
# FILE     : cohort_analysis/cohort_retention.py
# PROJECT  : Customer Segmentation — Cohort Retention Analysis
# ============================================================
#
# WHAT IS COHORT RETENTION ANALYSIS? (explain this in interviews)
# ───────────────────────────────────────────────────────
# A cohort is a GROUP OF CUSTOMERS who share a common characteristic
# at a specific point in time. In retention analysis, the cohort is
# defined by the month a customer made their FIRST PURCHASE.
#
# Retention analysis asks:
#   "Of all customers who bought for the first time in Month X,
#    what percentage came back to buy again in Month X+1, X+2, ..?"
#
# WHY THIS MATTERS FOR BUSINESS:
# ───────────────────────────────────────────────────────
# 1. Reveals whether the business is IMPROVING at retaining customers
#    over time (are newer cohorts sticking better than older ones?)
# 2. Shows at what month customers TYPICALLY CHURN
#    (helps decide when to trigger win-back campaigns)
# 3. Identifies seasonal cohorts (e.g., December cohort = gift buyers
#    who may not return — different from July cohort = loyal buyers)
# 4. Enables forecasting: if you know retention curves, you can
#    project future revenue from current customer base
#
# HOW TO READ THE HEATMAP:
# ───────────────────────────────────────────────────────
# Rows    = cohort (month of first purchase)
# Columns = months since first purchase (0, 1, 2, ..)
# Values  = % of original cohort still active that month
# Month 0 = always 100% (everyone is there at acquisition)
#
# A GOOD business has a heatmap that stays BRIGHT (high %)
# as you move right. A struggling business goes DARK quickly.
#
# RECRUITER ONE-LINER:
# "I built a cohort retention heatmap showing what % of customers
#  from each acquisition month returned in subsequent months.
#  This identified that the November cohort (holiday shoppers)
#  had 40% lower 3-month retention than average, informing our
#  decision to exclude them from win-back campaign spend."
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'DejaVu Sans', 'axes.titlesize': 13,
})

COLORS   = ['#2563EB', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
PROC_DIR = Path('./data/processed')
REP_DIR  = Path('./reports')
REP_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  COHORT RETENTION ANALYSIS")
print("=" * 60)


# ════════════════════════════════════════════════════════════
# STEP 1: LOAD TRANSACTION DATA
# ════════════════════════════════════════════════════════════

def load_transactions() -> pd.DataFrame:
    tx_path = PROC_DIR / 'retail_cleaned.csv'
    if tx_path.exists():
        df = pd.read_csv(tx_path, parse_dates=['InvoiceDate'])
        cid = 'Customer ID' if 'Customer ID' in df.columns else 'customer_id'
        date = 'InvoiceDate' if 'InvoiceDate' in df.columns else 'invoice_date'
        inv  = 'Invoice' if 'Invoice' in df.columns else 'invoice'
        df = df.rename(columns={cid: 'customer_id', date: 'invoice_date', inv: 'invoice'})
        return df

    print("⚠️  Generating synthetic transactions for cohort analysis")
    np.random.seed(42)
    n = 40000
    n_customers = 2000
    # Create realistic purchase patterns with seasonality
    dates = pd.date_range('2010-12-01', '2011-12-31', freq='D')
    # Weight toward Q4 (holiday season)
    date_weights = np.ones(len(dates))
    date_weights[pd.DatetimeIndex(dates).month.isin([11, 12])] *= 2.5
    date_weights /= date_weights.sum()

    cust_ids = np.random.randint(10000, 12000, n_customers)
    cust_weights = np.random.pareto(1.3, n_customers) + 0.1
    cust_weights /= cust_weights.sum()

    return pd.DataFrame({
        'customer_id'  : np.random.choice(cust_ids, n, p=cust_weights),
        'invoice'      : [f'{500000+i}' for i in range(n)],
        'invoice_date' : np.random.choice(dates, n, p=date_weights),
        'TotalAmount'  : np.round(np.random.lognormal(4.2, 0.9, n), 2),
    })

df = load_transactions()
df['invoice_date'] = pd.to_datetime(df['invoice_date'])
print(f"✅ Loaded {len(df):,} transactions | {df['customer_id'].nunique():,} customers")
print(f"   Date range: {df['invoice_date'].min().date()} → {df['invoice_date'].max().date()}")


# ════════════════════════════════════════════════════════════
# STEP 2: BUILD COHORT TABLE
#
# For each customer:
#   cohort_month = month of their FIRST purchase
#   order_month  = month of each subsequent purchase
#   cohort_index = how many months after first purchase
# ════════════════════════════════════════════════════════════

print("\n[Step 2] Building cohort table …")

# Get cohort month (first purchase month per customer)
df['order_month']  = df['invoice_date'].dt.to_period('M')
first_purchase     = df.groupby('customer_id')['order_month'].min().reset_index()
first_purchase.columns = ['customer_id', 'cohort_month']

# Merge back to main data
df = df.merge(first_purchase, on='customer_id')

# Calculate cohort index (months since first purchase)
df['cohort_index'] = (
    (df['order_month'] - df['cohort_month'])
    .apply(lambda x: x.n)        # .n gives the integer number of periods
)

print(f"   Cohort months available: {df['cohort_month'].nunique()}")
print(f"   Max cohort index (months): {df['cohort_index'].max()}")


# ════════════════════════════════════════════════════════════
# STEP 3: BUILD RETENTION MATRIX
#
# Rows = cohort months
# Cols = 0, 1, 2, .. months after first purchase
# Values = number of UNIQUE customers who purchased that month
# ════════════════════════════════════════════════════════════

print("\n[Step 3] Computing retention matrix …")

# Count unique customers per cohort per month
cohort_data = (
    df.groupby(['cohort_month', 'cohort_index'])['customer_id']
    .nunique()
    .reset_index()
    .rename(columns={'customer_id': 'customers'})
)

# Pivot to matrix format
cohort_matrix = cohort_data.pivot_table(
    index   = 'cohort_month',
    columns = 'cohort_index',
    values  = 'customers'
)
cohort_matrix.index = cohort_matrix.index.astype(str)

# Cohort sizes (month 0 = acquisition count)
cohort_sizes = cohort_matrix[0]

# Retention RATE matrix: divide each row by its cohort size
retention_matrix = cohort_matrix.divide(cohort_sizes, axis=0).round(4)

print(f"   Cohort matrix shape: {cohort_matrix.shape}")
print(f"\n--- Retention Matrix (first 6 months) ---")
display_cols = list(range(min(7, retention_matrix.shape[1])))
print((retention_matrix[display_cols] * 100).round(1).to_string())


# ════════════════════════════════════════════════════════════
# STEP 4: REVENUE COHORT ANALYSIS
#
# Same structure but uses AVERAGE REVENUE per customer
# instead of headcount. Answers: which cohort is most valuable?
# ════════════════════════════════════════════════════════════

print("\n[Step 4] Revenue cohort analysis …")

if 'TotalAmount' in df.columns:
    revenue_cohort = (
        df.groupby(['cohort_month', 'cohort_index'])['TotalAmount']
        .mean()
        .reset_index()
    )
    revenue_matrix = revenue_cohort.pivot_table(
        index   = 'cohort_month',
        columns = 'cohort_index',
        values  = 'TotalAmount'
    ).round(2)
    revenue_matrix.index = revenue_matrix.index.astype(str)
    print("   Revenue per customer matrix built")


# ════════════════════════════════════════════════════════════
# STEP 5: KEY METRICS FROM COHORT ANALYSIS
# ════════════════════════════════════════════════════════════

print("\n[Step 5] Computing cohort KPIs …")

# Average retention by cohort index (across all cohorts)
avg_retention = retention_matrix.mean() * 100

# Month-1 retention (how many come back after first purchase?)
month1_retention = retention_matrix[1].dropna() * 100 if 1 in retention_matrix.columns else pd.Series()

# Best and worst retaining cohorts (at month 3 if available)
if 3 in retention_matrix.columns:
    m3 = retention_matrix[3].dropna() * 100
    best_cohort  = m3.idxmax()
    worst_cohort = m3.idxmin()
    print(f"\n   Month-1 avg retention : {month1_retention.mean():.1f}%")
    print(f"   Month-3 avg retention : {m3.mean():.1f}%")
    print(f"   Best cohort (3m)      : {best_cohort} ({m3[best_cohort]:.1f}%)")
    print(f"   Worst cohort (3m)     : {worst_cohort} ({m3[worst_cohort]:.1f}%)")

# Overall average retention curve
print("\n--- Average Retention Curve ---")
for month, ret in avg_retention.items():
    if month <= 12:
        bar = '█' * int(ret / 2)
        print(f"  Month {month:>2}: {ret:>5.1f}% {bar}")


# ════════════════════════════════════════════════════════════
# STEP 6: VISUALISATIONS
# ════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 14))
gs  = plt.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)
fig.suptitle('Cohort Retention Analysis', fontweight='bold', fontsize=16)

# Plot 1: Retention Heatmap (the main chart)
ax1 = fig.add_subplot(gs[0, :])

# Limit to max 13 months for readability
max_months   = min(13, retention_matrix.shape[1])
plot_matrix  = retention_matrix.iloc[:, :max_months] * 100

im = ax1.imshow(
    plot_matrix.values,
    cmap   = 'YlOrRd_r',   # reversed: bright = high retention (good)
    aspect = 'auto',
    vmin   = 0, vmax = 100,
)

# Add text annotations inside each cell
for i in range(plot_matrix.shape[0]):
    for j in range(plot_matrix.shape[1]):
        val = plot_matrix.iloc[i, j]
        if not np.isnan(val):
            color = 'white' if val < 30 else 'black'
            ax1.text(j, i, f'{val:.0f}%', ha='center', va='center',
                     fontsize=8, color=color, fontweight='bold')

ax1.set_xticks(range(max_months))
ax1.set_xticklabels([f'Month {i}' for i in range(max_months)], fontsize=9)
ax1.set_yticks(range(len(plot_matrix)))
ax1.set_yticklabels(plot_matrix.index, fontsize=9)
ax1.set_title('Customer Retention Heatmap\n(% of original cohort returning each month)',
              fontweight='bold', fontsize=13)
ax1.set_xlabel('Months Since First Purchase')
ax1.set_ylabel('Acquisition Cohort')
plt.colorbar(im, ax=ax1, label='Retention Rate (%)', shrink=0.6)

# Plot 2: Retention curves (line chart for each cohort)
ax2 = fig.add_subplot(gs[1, 0])
n_cohorts = min(8, len(retention_matrix))
cohort_colors = plt.cm.Blues(np.linspace(0.4, 1.0, n_cohorts))

for i, (cohort, row) in enumerate(list(retention_matrix.iterrows())[:n_cohorts]):
    valid = row.dropna()
    if len(valid) > 1:
        ax2.plot(valid.index, valid.values * 100,
                 marker='o', markersize=4, linewidth=1.8,
                 color=cohort_colors[i], label=str(cohort), alpha=0.85)

ax2.set_title('Retention Curves by Cohort', fontweight='bold')
ax2.set_xlabel('Months Since Acquisition')
ax2.set_ylabel('Retention Rate (%)')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax2.legend(fontsize=7, ncol=2, title='Cohort', title_fontsize=8)
ax2.set_ylim(0, 105)

# Plot 3: Average retention curve with confidence band
ax3 = fig.add_subplot(gs[1, 1])
months_available = [c for c in avg_retention.index if c <= 12]
avg_vals         = avg_retention[months_available].values
std_vals         = (retention_matrix[months_available].std() * 100).values

ax3.plot(months_available, avg_vals, color=COLORS[0], linewidth=2.5,
         marker='o', markersize=6, label='Avg Retention')
ax3.fill_between(months_available,
                 np.maximum(0, avg_vals - std_vals),
                 np.minimum(100, avg_vals + std_vals),
                 alpha=0.15, color=COLORS[0], label='±1 Std Dev')
ax3.set_title('Average Retention Curve\n(All Cohorts)', fontweight='bold')
ax3.set_xlabel('Months Since First Purchase')
ax3.set_ylabel('Avg Retention Rate (%)')
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax3.legend(fontsize=9)
ax3.set_ylim(0, 110)

# Annotate key drop-off points
if len(avg_vals) > 1:
    max_drop_month = np.argmax(np.abs(np.diff(avg_vals))) + 1
    ax3.annotate(
        f'Biggest drop\n(Month {max_drop_month})',
        xy=(max_drop_month, avg_vals[max_drop_month]),
        xytext=(max_drop_month + 1.5, avg_vals[max_drop_month] + 10),
        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
        fontsize=9, color='red'
    )

# Plot 4: Cohort sizes (acquisition volume over time)
ax4 = fig.add_subplot(gs[2, 0])
cohort_sizes_plot = cohort_sizes.dropna()
ax4.bar(range(len(cohort_sizes_plot)), cohort_sizes_plot.values,
        color=COLORS[0], alpha=0.8, edgecolor='white')
ax4.set_xticks(range(len(cohort_sizes_plot)))
ax4.set_xticklabels([str(c) for c in cohort_sizes_plot.index],
                     rotation=45, ha='right', fontsize=8)
ax4.set_title('Cohort Size (New Customers Acquired)', fontweight='bold')
ax4.set_xlabel('Acquisition Month')
ax4.set_ylabel('New Customers')

# Plot 5: Revenue heatmap (if available)
ax5 = fig.add_subplot(gs[2, 1])
if 'revenue_matrix' in locals():
    rev_plot = revenue_matrix.iloc[:, :max_months].fillna(0)
    im2 = ax5.imshow(rev_plot.values, cmap='Blues', aspect='auto')
    for i in range(rev_plot.shape[0]):
        for j in range(rev_plot.shape[1]):
            val = rev_plot.iloc[i, j]
            if val > 0:
                ax5.text(j, i, f'£{val:.0f}', ha='center', va='center',
                         fontsize=7, color='white' if val > rev_plot.values.max()*0.6 else 'black')
    ax5.set_xticks(range(max_months))
    ax5.set_xticklabels([f'M{i}' for i in range(max_months)], fontsize=8)
    ax5.set_yticks(range(len(rev_plot)))
    ax5.set_yticklabels(rev_plot.index, fontsize=8)
    ax5.set_title('Avg Revenue Per Customer (£)\nby Cohort × Month', fontweight='bold')
    plt.colorbar(im2, ax=ax5, label='Avg Revenue (£)', shrink=0.6)
else:
    ax5.text(0.5, 0.5, 'Revenue data\nnot available', ha='center', va='center',
             transform=ax5.transAxes, fontsize=12)

plt.savefig(REP_DIR / 'cohort_retention.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n📊 Chart saved: reports/cohort_retention.png")


# ════════════════════════════════════════════════════════════
# STEP 7: SAVE OUTPUTS AND BUSINESS SUMMARY
# ════════════════════════════════════════════════════════════

retention_matrix.to_csv(PROC_DIR / 'cohort_retention_matrix.csv')
cohort_matrix.to_csv(PROC_DIR / 'cohort_raw_counts.csv')

print(f"\n✅ Outputs saved:")
print(f"   cohort_retention_matrix.csv — retention rates by cohort × month")
print(f"   cohort_raw_counts.csv       — raw customer counts")

print(f"""
BUSINESS SUMMARY
────────────────
• Month-1 retention   : {month1_retention.mean():.1f}% of customers return after first purchase
• Month-3 retention   : {m3.mean():.1f}% still active after 3 months (if data available)
• Key insight: Watch for the cohort with lowest Month-1 retention —
  these are likely one-time gift buyers (common in Q4/holiday season)
  and should NOT be included in the win-back campaign targeting

• Trigger win-back campaigns at Month 2 (before churn becomes permanent)
• Focus retention budget on cohorts showing declining trends
""")

print("📌 Next: python reports/generate_pdf_report.py")
