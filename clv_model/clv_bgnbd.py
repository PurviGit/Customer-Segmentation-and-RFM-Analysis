# ============================================================
# FILE     : clv_model/clv_bgnbd.py
# PROJECT  : Customer Segmentation — RFM + CLV Prediction
# ============================================================
#
# WHAT IS THE BG/NBD MODEL? (explain this in interviews)
# ───────────────────────────────────────────────────────
# BG/NBD stands for Beta-Geometric / Negative Binomial Distribution.
# It is a PROBABILISTIC model — not a black-box ML model — that
# models two simultaneous processes for each customer:
#
#   1. PURCHASE PROCESS (Negative Binomial):
#      While a customer is "alive" (still interested in the brand),
#      they make purchases at some personal rate λ (lambda).
#      This rate varies across customers — some buy weekly,
#      some buy monthly. The NB distribution captures this variation.
#
#   2. DROPOUT PROCESS (Beta-Geometric):
#      After any purchase, a customer may "die" (churn) with
#      some probability p. This also varies across customers.
#      The Beta distribution captures this variation.
#
# WHY IS THIS BETTER THAN OUR FORMULA? (the key interview answer)
# ───────────────────────────────────────────────────────
# Our previous CLV estimate was:  (Monetary / Recency) × 365
# This is a RATIO — it doesn't account for:
#   - The probability that the customer is still active
#   - Individual variation in purchase rates
#   - The fact that a customer who bought once 2 years ago
#     is NOT the same CLV risk as one who bought once last week
#
# BG/NBD gives us: P(alive) × Expected future purchases × Avg order value
# This is the formula used by Amazon, Spotify, Booking.com.
#
# Developed by Peter Fader (Wharton) — widely published, peer-reviewed.
#
# RECRUITER ONE-LINER:
# "I used the BG/NBD probabilistic model to predict how many purchases
#  each customer will make in the next 12 months and their probability
#  of still being an active customer — the same approach used by
#  subscription and e-commerce companies to prioritise retention spend."
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Try importing lifetimes library
try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    from lifetimes.plotting import (
        plot_frequency_recency_matrix,
        plot_probability_alive_matrix,
        plot_period_transactions,
    )
    from lifetimes.utils import summary_data_from_transaction_data
    LIFETIMES_AVAILABLE = True
except ImportError:
    LIFETIMES_AVAILABLE = False
    print("⚠️  lifetimes not installed. Run: pip install lifetimes")
    print("   Falling back to manual BG/NBD approximation.\n")

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#f8f9fa',
    'axes.grid': True, 'grid.alpha': 0.4,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'DejaVu Sans', 'axes.titlesize': 13, 'axes.labelsize': 11,
})

COLORS   = ['#2563EB', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
PROC_DIR = Path('./data/processed')
REP_DIR  = Path('./reports')
REP_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  CUSTOMER LIFETIME VALUE — BG/NBD + Gamma-Gamma Model")
print("=" * 60)


# ════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ════════════════════════════════════════════════════════════

def load_transaction_data() -> pd.DataFrame:
    """Load cleaned transaction data."""
    paths = [
        PROC_DIR / 'retail_cleaned.csv',
        PROC_DIR / 'rfm_final.csv',
    ]
    for p in paths:
        if p.exists():
            df = pd.read_csv(p, parse_dates=['InvoiceDate'] if 'InvoiceDate' in pd.read_csv(p, nrows=0).columns else [])
            print(f"✅ Loaded {p.name}: {len(df):,} rows")
            return df

    # Fallback: generate synthetic transaction data
    print("⚠️  No cleaned data found — generating synthetic transactions")
    np.random.seed(42)
    n = 30000
    n_customers = 1500
    rng = pd.date_range('2010-12-01', '2011-12-09', freq='H')

    # Pareto customer weights (20% buy 80% of the time)
    cust_ids = np.arange(10000, 10000 + n_customers)
    weights  = np.random.pareto(1.2, n_customers) + 0.1
    weights /= weights.sum()

    return pd.DataFrame({
        'Customer ID'  : np.random.choice(cust_ids, n, p=weights),
        'Invoice'      : [f'{500000+i}' for i in range(n)],
        'InvoiceDate'  : pd.to_datetime(np.random.choice(rng, n)),
        'TotalAmount'  : np.round(np.random.lognormal(4.0, 0.9, n), 2).clip(1, 2000),
    })

df_raw = load_transaction_data()

# Ensure correct column names
if 'InvoiceDate' not in df_raw.columns and 'invoice_date' in df_raw.columns:
    df_raw.rename(columns={'invoice_date': 'InvoiceDate', 'customer_id': 'Customer ID',
                            'invoice': 'Invoice', 'total_amount': 'TotalAmount'}, inplace=True)

if 'TotalAmount' not in df_raw.columns and 'Monetary' in df_raw.columns:
    # We have RFM data, not transaction data — reconstruct approximate transactions
    print("  Using RFM data to approximate transaction history")
    rfm = df_raw.copy()
    df_raw = None
else:
    rfm = None

df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate'], errors='coerce')
df_raw.dropna(subset=['InvoiceDate', 'Customer ID'], inplace=True)

observation_end = df_raw['InvoiceDate'].max()
print(f"   Observation period: {df_raw['InvoiceDate'].min().date()} → {observation_end.date()}")
print(f"   Unique customers  : {df_raw['Customer ID'].nunique():,}")


# ════════════════════════════════════════════════════════════
# STEP 2: BUILD RFM SUMMARY TABLE FOR LIFETIMES LIBRARY
#
# The lifetimes library needs a specific format:
#   frequency : number of REPEAT purchases (total - 1)
#   recency   : time between first and LAST purchase (weeks)
#   T         : age of the customer = time since first purchase (weeks)
#   monetary_value: average transaction value (for Gamma-Gamma)
# ════════════════════════════════════════════════════════════

if LIFETIMES_AVAILABLE and df_raw is not None:
    print("\n[Step 2] Building BG/NBD input table …")
    rfm_summary = summary_data_from_transaction_data(
        df_raw,
        customer_id_col    = 'Customer ID',
        datetime_col       = 'InvoiceDate',
        monetary_value_col = 'TotalAmount',
        observation_period_end = observation_end,
        freq               = 'W',          # weekly time units
    )
    # Remove one-time buyers from Gamma-Gamma (needs repeat purchasers)
    rfm_gg = rfm_summary[rfm_summary['frequency'] > 0].copy()

    print(f"   Total customers          : {len(rfm_summary):,}")
    print(f"   Repeat purchasers        : {len(rfm_gg):,} ({len(rfm_gg)/len(rfm_summary)*100:.1f}%)")
    print(f"   One-time buyers          : {len(rfm_summary) - len(rfm_gg):,}")
    print(f"\n--- BG/NBD Input Table Sample ---")
    print(rfm_summary.head(5).round(2).to_string())

else:
    # Manual fallback: build the summary from RFM data if lifetimes not available
    print("\n⚠️  Building approximate CLV without lifetimes library")
    if rfm is None:
        rfm = pd.read_csv(PROC_DIR / 'rfm_scored.csv') if (PROC_DIR / 'rfm_scored.csv').exists() else None

    if rfm is not None:
        rfm_summary = pd.DataFrame({
            'customer_id'    : rfm['Customer ID'] if 'Customer ID' in rfm.columns else rfm.index,
            'frequency'      : rfm['Frequency'] - 1,        # repeat purchases
            'recency'        : (365 - rfm['Recency']) / 7,  # approximate weeks
            'T'              : 52,                           # assume 1-year observation
            'monetary_value' : rfm['Monetary'] / rfm['Frequency'],
        })
        rfm_summary = rfm_summary[rfm_summary['frequency'] >= 0]
        rfm_gg      = rfm_summary[rfm_summary['frequency'] > 0]
        LIFETIMES_AVAILABLE = False
    else:
        print("❌ No data available. Run notebooks first.")
        exit(1)


# ════════════════════════════════════════════════════════════
# STEP 3: FIT BG/NBD MODEL (purchase frequency prediction)
#
# The model learns 4 parameters from data:
#   r, alpha : shape the distribution of purchase rates
#   a, b     : shape the distribution of dropout probabilities
#
# Once fit, we can ask: "For customer X with these purchase
# patterns, how many purchases will they make in next N weeks?"
# ════════════════════════════════════════════════════════════

print("\n[Step 3] Fitting BG/NBD model …")

if LIFETIMES_AVAILABLE:
    bgf = BetaGeoFitter(
    penalizer_coef=1.0
)
    bgf.fit(
        rfm_summary['frequency'],
        rfm_summary['recency'],
        rfm_summary['T'],
    )

    print(f"✅ BG/NBD model fitted")
    print(f"   Parameters: r={bgf.params_['r']:.4f}, alpha={bgf.params_['alpha']:.4f}, "
          f"a={bgf.params_['a']:.4f}, b={bgf.params_['b']:.4f}")
    if hasattr(bgf, 'log_likelihood_'):
        print(f"   Log-likelihood: {bgf.log_likelihood_:.2f}")


    # Predict expected purchases in next 52 weeks (1 year)
    t_prediction = 52  # weeks

    rfm_summary['predicted_purchases_12m'] = bgf.conditional_expected_number_of_purchases_up_to_time(
        t_prediction,
        rfm_summary['frequency'],
        rfm_summary['recency'],
        rfm_summary['T'],
    ).round(2)

    rfm_summary['prob_alive'] = bgf.conditional_probability_alive(
        rfm_summary['frequency'],
        rfm_summary['recency'],
        rfm_summary['T'],
    ).round(4)

else:
    # Manual approximation when lifetimes not available
    # Uses a simplified Pareto/NBD approximation
    freq = rfm_summary['frequency'].values
    rec  = rfm_summary['recency'].values
    T    = rfm_summary['T'].values

    # Approximate P(alive): higher recency relative to T = more likely alive
    rfm_summary['prob_alive'] = np.where(
        freq == 0,
        0.30,
        np.clip(rec / T * (1 + freq * 0.1), 0, 0.99)
    ).round(4)

    # Approximate predicted purchases
    rfm_summary['predicted_purchases_12m'] = (
        rfm_summary['prob_alive'] * (freq + 1) * (52 / T.clip(1))
    ).round(2)

print(f"\n--- Predicted Purchases (next 12 months) ---")
print(rfm_summary[['frequency','recency','predicted_purchases_12m','prob_alive']].describe().round(2))


# ════════════════════════════════════════════════════════════
# STEP 4: FIT GAMMA-GAMMA MODEL (monetary value prediction)
#
# The Gamma-Gamma model predicts the AVERAGE ORDER VALUE
# for future transactions. It assumes:
# - Each transaction value is independent and identically distributed
# - Customer-level average spend varies across customers (Gamma dist)
#
# IMPORTANT: This model is only meaningful for repeat purchasers
# (customers with frequency > 0). We filtered to rfm_gg for this.
# ════════════════════════════════════════════════════════════

print("\n[Step 4] Fitting Gamma-Gamma model …")

if LIFETIMES_AVAILABLE:
    # Check correlation between frequency and monetary
    # (Gamma-Gamma assumes independence — correlation should be low)
    corr = rfm_gg[['frequency','monetary_value']].corr().iloc[0,1]
    print(f"   Frequency-Monetary correlation: {corr:.4f}  (should be < 0.3)")

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(rfm_gg['frequency'], rfm_gg['monetary_value'])

    print(f"✅ Gamma-Gamma model fitted")
    print(f"   Parameters: p={ggf.params_['p']:.4f}, q={ggf.params_['q']:.4f}, "
          f"v={ggf.params_['v']:.4f}")

    # Predicted average transaction value per customer
    rfm_gg['predicted_avg_order'] = ggf.conditional_expected_average_profit(
        rfm_gg['frequency'],
        rfm_gg['monetary_value'],
    ).round(2)

    # Full CLV prediction: purchases × value × margin × discount_rate
    DISCOUNT_RATE  = 0.10   # 10% annual discount rate (cost of capital)
    GROSS_MARGIN   = 0.65   # 65% gross margin (typical retail)
    TIME_HORIZON   = 12     # months

    rfm_gg['clv_predicted'] = ggf.customer_lifetime_value(
        bgf,
        rfm_gg['frequency'],
        rfm_gg['recency'],
        rfm_gg['T'],
        rfm_gg['monetary_value'],
        time            = TIME_HORIZON,
        discount_rate   = DISCOUNT_RATE / 12,  # monthly
        freq            = 'W',
    ).round(2)

else:
    # Manual CLV approximation
    rfm_gg['predicted_avg_order'] = rfm_gg['monetary_value'] * 1.05  # slight regression to mean
    rfm_gg['clv_predicted'] = (
        rfm_gg['predicted_purchases_12m'] *
        rfm_gg['predicted_avg_order'] *
        0.65  # gross margin
    ).round(2)

print(f"\n--- CLV Predictions ---")
print(rfm_gg[['monetary_value','predicted_avg_order','clv_predicted']].describe().round(2))


# ════════════════════════════════════════════════════════════
# STEP 5: COMBINE WITH SEGMENTS AND BUILD FINAL TABLE
# ════════════════════════════════════════════════════════════

print("\n[Step 5] Building final CLV table …")

# Load segment labels
seg_path = PROC_DIR / 'rfm_final.csv'
if seg_path.exists():
    seg_df = pd.read_csv(seg_path)
    cid_col = 'Customer ID' if 'Customer ID' in seg_df.columns else 'customer_id'
    seg_col = 'Segment' if 'Segment' in seg_df.columns else 'segment'

    # Merge CLV predictions with segment labels
    # Merge CLV predictions with segment labels

    rfm_gg_reset = rfm_gg.reset_index()

# Bring BG/NBD prediction columns into final table
    prediction_cols = rfm_summary.reset_index()[[
     'Customer ID',
     'predicted_purchases_12m',
     'prob_alive'
    ]]

    rfm_gg_reset = rfm_gg_reset.merge(
     prediction_cols,
     on='Customer ID',
     how='left'
    )

    clv_final = rfm_gg_reset.merge(
        seg_df[[cid_col, seg_col]].rename(
         columns={
            cid_col: 'Customer ID',
             seg_col: 'Segment'
         }
        ),
        on='Customer ID',
        how='left'
   )
else:
    clv_final = rfm_gg.reset_index() if rfm_gg.index.name else rfm_gg.copy()
    clv_final['Segment'] = 'Unknown'

print(f"✅ Final CLV table: {len(clv_final):,} customers with predictions")

# CLV tiers
clv_final['CLV_Tier'] = pd.qcut(
    clv_final['clv_predicted'].clip(lower=0),
    q=4,
    labels=['Bronze', 'Silver', 'Gold', 'Platinum'],
    duplicates='drop'
)

print("\n--- CLV by Segment ---")
if 'Segment' in clv_final.columns:
    seg_clv = clv_final.groupby('Segment').agg(
        Customers       = ('Customer ID', 'count'),
        Avg_CLV         = ('clv_predicted',  'mean'),
        Total_CLV       = ('clv_predicted',  'sum'),
        Avg_Prob_Alive  = ('prob_alive',      'mean'),
        Avg_Pred_Orders = ('predicted_purchases_12m', 'mean'),
    ).round(2)
    print(seg_clv.to_string())


# ════════════════════════════════════════════════════════════
# STEP 6: VISUALISATIONS
# ════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Customer Lifetime Value — BG/NBD + Gamma-Gamma Model',
             fontweight='bold', fontsize=15)

# Plot 1: CLV distribution
ax = axes[0, 0]
clv_vals = clv_final['clv_predicted'].clip(0, clv_final['clv_predicted'].quantile(0.98))
ax.hist(clv_vals, bins=50, color=COLORS[0], alpha=0.8, edgecolor='white')
ax.axvline(clv_vals.mean(),   color=COLORS[3], linestyle='--', lw=2, label=f"Mean: £{clv_vals.mean():.0f}")
ax.axvline(clv_vals.median(), color=COLORS[2], linestyle='--', lw=2, label=f"Median: £{clv_vals.median():.0f}")
ax.set_title('CLV Distribution (12-Month)', fontweight='bold')
ax.set_xlabel('Predicted CLV (£)')
ax.set_ylabel('Customers')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x:.0f}'))
ax.legend(fontsize=9)

# Plot 2: P(alive) distribution
ax = axes[0, 1]
ax.hist(clv_final['prob_alive'], bins=40, color=COLORS[1], alpha=0.8, edgecolor='white')
ax.axvline(0.5, color='red', linestyle='--', lw=1.5, label='50% threshold')
ax.set_title('Probability Customer Still Active', fontweight='bold')
ax.set_xlabel('P(alive)')
ax.set_ylabel('Customers')
pct_alive = (clv_final['prob_alive'] >= 0.5).mean() * 100
ax.text(0.05, 0.9, f'{pct_alive:.0f}% likely still active',
        transform=ax.transAxes, fontsize=10, color=COLORS[1], fontweight='bold')
ax.legend(fontsize=9)

# Plot 3: Predicted purchases by segment
if 'Segment' in clv_final.columns and clv_final['Segment'].notna().any():
    ax = axes[0, 2]
    seg_orders = clv_final.groupby('Segment')['predicted_purchases_12m'].mean().sort_values()
    colors_bar = [COLORS[i % len(COLORS)] for i in range(len(seg_orders))]
    bars = ax.barh(seg_orders.index, seg_orders.values, color=colors_bar, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, seg_orders.values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=9, fontweight='bold')
    ax.set_title('Avg Predicted Orders (12m) by Segment', fontweight='bold')
    ax.set_xlabel('Expected Orders')

# Plot 4: CLV vs Frequency scatter
ax = axes[1, 0]
sample = clv_final.sample(min(500, len(clv_final)), random_state=42)
scatter = ax.scatter(sample['frequency'], sample['clv_predicted'],
                     c=sample['prob_alive'], cmap='RdYlGn',
                     alpha=0.6, s=20, vmin=0, vmax=1)
plt.colorbar(scatter, ax=ax, label='P(alive)')
ax.set_title('CLV vs Purchase Frequency\n(color = P(alive))', fontweight='bold')
ax.set_xlabel('Historical Frequency (orders)')
ax.set_ylabel('Predicted CLV 12m (£)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x:.0f}'))

# Plot 5: CLV by tier bar chart
ax = axes[1, 1]
if 'CLV_Tier' in clv_final.columns:
    tier_data = clv_final.groupby('CLV_Tier').agg(
        Customers = ('Customer ID', 'count'),
        Avg_CLV   = ('clv_predicted', 'mean'),
    ).reset_index()
    tier_colors = ['#CD7F32', '#C0C0C0', '#FFD700', '#E5E4E2']
    bars = ax.bar(tier_data['CLV_Tier'], tier_data['Avg_CLV'],
                  color=tier_colors[:len(tier_data)], alpha=0.85, edgecolor='white')
    for bar, row in zip(bars, tier_data.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'£{row.Avg_CLV:.0f}\n({row.Customers} cust.)',
                ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Average CLV by Tier', fontweight='bold')
    ax.set_xlabel('CLV Tier')
    ax.set_ylabel('Avg Predicted CLV (£)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x:.0f}'))

# Plot 6: P(alive) vs Recency heatmap proxy
ax = axes[1, 2]
if LIFETIMES_AVAILABLE:
    try:
        plot_probability_alive_matrix(bgf, ax=ax)
        ax.set_title('P(Alive) Matrix\n(Frequency vs Recency)', fontweight='bold')
    except:
        ax.text(0.5, 0.5, 'P(alive) matrix\nrequires lifetimes>=0.11',
                ha='center', va='center', transform=ax.transAxes)
else:
    # Manual heatmap
    freq_bins = [0, 1, 3, 5, 10, 30]
    rec_bins  = [0, 10, 20, 30, 40, 52]
    clv_final['f_bin'] = pd.cut(clv_final['frequency'], bins=freq_bins)
    clv_final['r_bin'] = pd.cut(clv_final['recency'],   bins=rec_bins)
    pivot = clv_final.groupby(['f_bin','r_bin'])['prob_alive'].mean().unstack()
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_title('Avg P(alive): Frequency vs Recency', fontweight='bold')
    plt.colorbar(im, ax=ax, label='P(alive)')

plt.tight_layout()
plt.savefig(REP_DIR / 'clv_model_results.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n📊 Chart saved: reports/clv_model_results.png")


# ════════════════════════════════════════════════════════════
# STEP 7: BUSINESS OUTPUT
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  BUSINESS INSIGHTS FROM CLV MODEL")
print("=" * 60)

total_predicted_clv = clv_final['clv_predicted'].sum()
likely_active       = (clv_final['prob_alive'] >= 0.5).sum()
top10_clv           = clv_final.nlargest(int(len(clv_final)*0.10), 'clv_predicted')['clv_predicted'].sum()

print(f"""
  Total predicted 12m CLV (active customers): £{total_predicted_clv:,.0f}
  Customers likely still active (P≥0.5)     : {likely_active:,} of {len(clv_final):,}
  Top 10% customers' CLV                    : £{top10_clv:,.0f}
  Top 10% share of total CLV                : {top10_clv/total_predicted_clv*100:.1f}%

  Platinum tier avg CLV : see CLV by Tier chart above
  Gold tier avg CLV     : see CLV by Tier chart above

  RECOMMENDATION:
  Allocate retention budget proportional to predicted CLV,
  not equally across all customers. The top 10% CLV customers
  warrant personal outreach; the bottom 30% may be unprofitable
  to retain at high cost.
""")

# Save output
clv_final.to_csv(PROC_DIR / 'clv_predictions.csv', index=False)
print(f"✅ CLV predictions saved: data/processed/clv_predictions.csv")
print("\n📌 Next: python reports/generate_pdf_report.py")
