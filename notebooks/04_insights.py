# ============================================================
# NOTEBOOK 04 — Business Insights & Marketing Strategy
# Project : Customer Segmentation using RFM + Clustering
# ============================================================
#
# WHAT THIS NOTEBOOK DOES (for recruiter reference):
# ─────────────────────────────────────────────────
# 1. Translates ML cluster output into actionable business language
# 2. Builds a full segment strategy table (channel, message, offer)
# 3. Estimates revenue opportunity per segment
# 4. Creates a churn risk assessment
# 5. Produces executive-ready charts and a summary table
# 6. Exports a Power BI-compatible Excel file and Tableau CSV
#
# WHY THIS MATTERS TO RECRUITERS:
# ─────────────────────────────────────────────────
# Most candidates can run K-Means. Few can then answer:
# "So what should the marketing team actually DO?"
# This notebook shows you bridge data science → business value.
# ============================================================

# %% — Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor':'white','axes.facecolor':'#f8f9fa',
    'axes.grid':True,'grid.alpha':0.4,
    'axes.spines.top':False,'axes.spines.right':False,
    'font.family':'DejaVu Sans','axes.titlesize':13,'axes.labelsize':11,
})

COLORS   = ['#2563EB','#10B981','#F59E0B','#EF4444','#8B5CF6']
PROC_DIR = './data/processed/'
REP_DIR  = './reports/'
os.makedirs(REP_DIR, exist_ok=True)

# %% — Load data
rfm     = pd.read_csv(f'{PROC_DIR}rfm_final.csv')
summary = pd.read_csv(f'{PROC_DIR}segment_summary.csv')

print(f"✅ Data loaded: {rfm.shape}")
print(f"\nSegments present: {rfm['Segment'].unique().tolist()}")

SEGMENTS = rfm['Segment'].unique().tolist()

# %% [markdown]
# ## 1. Marketing Strategy by Segment
#
# For each segment, we define:
# - **Channel**: Where to reach them
# - **Message**: What to say
# - **Offer**: What incentive to use
# - **Goal**: What business outcome we're targeting
# - **KPI**: How we measure success

# %% — Strategy table
strategy = {
    'Champions': {
        'Emoji': '🏆',
        'Channel': 'Email + Loyalty App + VIP SMS',
        'Message': 'You are our top customer — get early access and exclusive rewards',
        'Offer': 'Loyalty points × 2, Early access to new products, Free shipping',
        'Goal': 'Retain & upsell — grow basket size',
        'KPI': 'Repeat purchase rate, Average Order Value',
        'Urgency': 'Low — they buy without prompting',
        'Budget_Allocation': '15% of marketing budget',
        'Expected_ROI': '5–8× spend',
        'Risk': 'Low churn risk — focus on maintaining relationship',
    },
    'Loyal Customers': {
        'Emoji': '⭐',
        'Channel': 'Email + Push notifications',
        'Message': 'Thank you for your loyalty — here is something special for you',
        'Offer': '10% loyalty discount, Referral programme, Birthday reward',
        'Goal': 'Move to Champions tier',
        'KPI': 'Purchase frequency increase, NPS score',
        'Urgency': 'Medium — nurture relationship',
        'Budget_Allocation': '25% of marketing budget',
        'Expected_ROI': '3–5× spend',
        'Risk': 'Medium — could slip to At Risk without engagement',
    },
    'Potential Loyal': {
        'Emoji': '🌱',
        'Channel': 'Email + Retargeting ads',
        'Message': 'We noticed you liked X — here are things you might love',
        'Offer': 'Free shipping on next order, 15% off second purchase',
        'Goal': 'Increase purchase frequency',
        'KPI': 'Second purchase rate, Email open rate',
        'Urgency': 'High — window of opportunity before they forget brand',
        'Budget_Allocation': '25% of marketing budget',
        'Expected_ROI': '2–3× spend',
        'Risk': 'Medium-high — haven\'t proven loyalty yet',
    },
    'At Risk': {
        'Emoji': '⚠️',
        'Channel': 'Win-back email sequence + Paid retargeting',
        'Message': 'We miss you! Here is an exclusive offer to come back',
        'Offer': '20% discount, Free gift with purchase, Extended return window',
        'Goal': 'Re-activate lapsed customers',
        'KPI': 'Reactivation rate, Revenue recovered',
        'Urgency': 'Very High — every week of inaction = lower recovery probability',
        'Budget_Allocation': '25% of marketing budget',
        'Expected_ROI': '1.5–2.5× spend',
        'Risk': 'High — act within 60 days or likely lost permanently',
    },
    'Lost / Inactive': {
        'Emoji': '💤',
        'Channel': 'Last-chance email, Survey',
        'Message': 'Is everything OK? We would love to know what went wrong',
        'Offer': '30% discount (last attempt), Feedback survey with incentive',
        'Goal': 'Minimal spend — identify reactivation vs write-off',
        'KPI': 'Response rate, Survey completion, Revenue per email sent',
        'Urgency': 'Low priority for spend — focus budget on higher segments',
        'Budget_Allocation': '10% of marketing budget',
        'Expected_ROI': '0.5–1.5× spend',
        'Risk': 'Very High — most will not reactivate',
    },
}

# Print strategy summary
for segment, strat in strategy.items():
    if segment in rfm['Segment'].unique():
        count = (rfm['Segment'] == segment).sum()
        rev   = rfm.loc[rfm['Segment'] == segment, 'Monetary'].sum()
        print(f"\n{'='*55}")
        print(f"  {strat['Emoji']} {segment.upper()} — {count:,} customers | £{rev:,.0f} revenue")
        print(f"{'='*55}")
        for k, v in strat.items():
            if k not in ['Emoji']:
                print(f"  {k:<22}: {v}")

# %% [markdown]
# ## 2. Revenue Opportunity Analysis
#
# We estimate the revenue uplift if each segment's Monetary value
# increases by a realistic percentage through the recommended strategy.

# %% — Revenue opportunity
opportunity = rfm.groupby('Segment').agg(
    Customers     = ('Customer ID', 'count'),
    Total_Revenue = ('Monetary',    'sum'),
    Avg_Revenue   = ('Monetary',    'mean'),
    Avg_CLV       = ('CLV_Estimate','mean'),
).reset_index()

uplift_targets = {
    'Champions'       : 0.10,
    'Loyal Customers' : 0.20,
    'Potential Loyal' : 0.35,
    'At Risk'         : 0.15,
    'Lost / Inactive' : 0.05,
}

opportunity['Uplift_Pct']         = opportunity['Segment'].map(uplift_targets).fillna(0.10)
opportunity['Revenue_Opportunity'] = (opportunity['Total_Revenue']
                                      * opportunity['Uplift_Pct']).round(0)

total_opportunity = opportunity['Revenue_Opportunity'].sum()
print(f"\n💰 Total Revenue Opportunity Estimate: £{total_opportunity:,.0f}")
print("\nBreakdown by segment:")
print(opportunity[['Segment','Customers','Total_Revenue','Uplift_Pct','Revenue_Opportunity']]
      .sort_values('Revenue_Opportunity', ascending=False)
      .to_string(index=False))

# %% — Revenue opportunity chart (Plotly waterfall)
fig = go.Figure(go.Waterfall(
    name='Revenue Uplift',
    orientation='v',
    measure=['relative'] * len(opportunity) + ['total'],
    x=list(opportunity['Segment']) + ['Total Opportunity'],
    y=list(opportunity['Revenue_Opportunity']) + [total_opportunity],
    text=[f"£{v:,.0f}" for v in list(opportunity['Revenue_Opportunity']) + [total_opportunity]],
    textposition='outside',
    connector={'line': {'color': '#94A3B8'}},
    increasing={'marker': {'color': '#10B981'}},
    totals={'marker': {'color': '#2563EB'}},
))
fig.update_layout(
    title='Revenue Opportunity by Segment (estimated uplift from strategy)',
    template='plotly_white',
    showlegend=False,
    height=450
)
fig.write_html(f'{REP_DIR}14_revenue_opportunity.html')
fig.show()
print(f"💾 Chart saved: {REP_DIR}14_revenue_opportunity.html")

# %% — Chart: Segment contribution (treemap)
fig_tree = px.treemap(
    rfm,
    path=['Segment'],
    values='Monetary',
    color='RFM_Score',
    color_continuous_scale='Blues',
    title='Revenue Contribution by Segment',
)
fig_tree.update_layout(template='plotly_white', height=450)
fig_tree.write_html(f'{REP_DIR}15_treemap.html')
fig_tree.show()
print(f"💾 Treemap saved: {REP_DIR}15_treemap.html")

# %% [markdown]
# ## 3. Churn Risk Analysis
#
# Churn probability is estimated from Recency:
# - Customers who haven't bought in >180 days: ~85% churn probability
# - 90–180 days: ~60%
# - 30–90 days: ~25%
# - <30 days: ~5%

# %% — Churn risk
def churn_probability(recency_days):
    if   recency_days > 180: return 0.85
    elif recency_days > 90:  return 0.60
    elif recency_days > 30:  return 0.25
    else:                    return 0.05

rfm['Churn_Probability'] = rfm['Recency'].apply(churn_probability)
rfm['Churn_Risk']        = pd.cut(rfm['Churn_Probability'],
                                   bins=[0, 0.1, 0.3, 0.65, 1.0],
                                   labels=['Low','Medium','High','Critical'])

churn_summary = rfm.groupby(['Segment','Churn_Risk']).size().reset_index(name='Count')
print("\n--- Churn Risk by Segment ---")
print(churn_summary.to_string(index=False))

at_risk_revenue = rfm.loc[rfm['Churn_Probability'] >= 0.60, 'Monetary'].sum()
print(f"\n⚠️  Revenue at High/Critical churn risk: £{at_risk_revenue:,.0f}")
print(f"   That is {at_risk_revenue/rfm['Monetary'].sum()*100:.1f}% of total historical revenue")

# %% [markdown]
# ## 4. Export for Power BI & Tableau

# %% — Power BI Excel export (multi-sheet)
pbi_path = f'{PROC_DIR}powerbi_data.xlsx'
with pd.ExcelWriter(pbi_path, engine='openpyxl') as writer:

    # Sheet 1: Full customer RFM table
    rfm.to_excel(writer, sheet_name='Customer_RFM', index=False)

    # Sheet 2: Segment summary
    summary.to_excel(writer, sheet_name='Segment_Summary', index=False)

    # Sheet 3: Strategy table
    strat_df = pd.DataFrame([
        {'Segment': k, **{k2: v2 for k2, v2 in v.items() if k2 != 'Emoji'}}
        for k, v in strategy.items()
    ])
    strat_df.to_excel(writer, sheet_name='Marketing_Strategy', index=False)

    # Sheet 4: Revenue opportunity
    opportunity.to_excel(writer, sheet_name='Revenue_Opportunity', index=False)

    # Sheet 5: Churn risk
    rfm[['Customer ID','Segment','Recency','Monetary','Churn_Probability','Churn_Risk']]\
        .to_excel(writer, sheet_name='Churn_Risk', index=False)
    
print(f"✅ Power BI Excel file saved: {pbi_path}")

# Tableau CSV (single flat table — Tableau works best with wide CSV)
tableau_path = f'{PROC_DIR}tableau_data.csv'
rfm.to_csv(tableau_path, index=False)
print(f"✅ Tableau CSV saved: {tableau_path}")
print(f"\n📌 Open powerbi_data.xlsx in Power BI Desktop → Get Data → Excel")
print(f"   Open tableau_data.csv in Tableau Public → Connect to Text File")
print("\n📌 Next: Run dashboard/app.py to launch Streamlit dashboard.")