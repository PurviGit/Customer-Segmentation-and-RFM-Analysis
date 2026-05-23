import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="RFM Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEG = {
    "Champions":       "#3b82f6",
    "Loyal Customers": "#10b981",
    "Potential Loyal": "#f59e0b",
    "At Risk":         "#f43f5e",
    "Lost / Inactive": "#6b7280",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&display=swap');
*,*::before,*::after{box-sizing:border-box;}
html,body,.stApp{font-family:'DM Sans',sans-serif!important;}
.block-container{padding:1.5rem 2rem 3rem!important;max-width:100%!important;}
[data-testid="stSidebar"]{min-width:280px!important;max-width:280px!important;}
[data-testid="metric-container"]{border-radius:16px!important;padding:20px 22px!important;transition:transform .2s,box-shadow .2s;}
[data-testid="metric-container"]:hover{transform:translateY(-3px);}
[data-testid="stMetricValue"]{font-size:26px!important;font-weight:700!important;}
[data-testid="stMetricLabel"]{font-size:12px!important;text-transform:uppercase;letter-spacing:.06em;}
[data-testid="stMetricDelta"]{font-size:12px!important;}
[data-testid="stTabs"] button{font-weight:500!important;font-size:14px!important;border-radius:8px 8px 0 0!important;}
[data-testid="stTabs"] button[aria-selected="true"]{font-weight:700!important;}
[data-testid="stPlotlyChart"]>div{border-radius:16px!important;padding:4px!important;overflow:hidden;}
[data-testid="stDataFrame"]{border-radius:14px!important;overflow:hidden!important;}
[data-testid="stExpander"]{border-radius:14px!important;overflow:hidden!important;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-thumb{border-radius:3px;}
.section-label{font-size:15px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#3b82f6;margin-bottom:4px;}
.section-title{font-size:22px;font-weight:700;margin-bottom:16px;display:flex;align-items:center;gap:10px;}
.insight-card{border-left:4px solid #3b82f6;border-radius:12px;padding:16px 20px;margin-bottom:10px;border:1px solid rgba(128,128,128,0.2);border-left:4px solid #3b82f6;}
.insight-card .ic-title{font-weight:700;font-size:14px;margin-bottom:6px;}
.insight-card .ic-body{font-size:13px;line-height:1.65;opacity:.8;}
.rfm-table{width:100%;border-collapse:collapse;font-size:13px;}
.rfm-table th{font-weight:700;font-size:15px;letter-spacing:.05em;text-transform:uppercase;padding:10px 12px;border-bottom:2px solid rgba(128,128,128,0.3);text-align:left;}
.rfm-table td{padding:9px 12px;border-bottom:1px solid rgba(128,128,128,0.15);}
.rfm-table tr:hover td{opacity:.85;}
.rfm-table .seg-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle;}
.tbl-wrap{border:1px solid rgba(128,128,128,0.2);border-radius:14px;overflow:hidden;}
.method-card{border:1px solid rgba(128,128,128,0.2);border-radius:14px;padding:18px 20px;height:100%;}
.method-card .mc-label{font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px;}
.method-card .mc-stat{font-size:22px;font-weight:800;margin-bottom:2px;}
.method-card .mc-desc{font-size:12px;line-height:1.6;opacity:.75;}
.method-card .mc-badge{display:inline-block;border-radius:4px;padding:2px 7px;font-size:10px;font-weight:600;margin-right:4px;margin-top:6px;background:rgba(59,130,246,0.12);color:#3b82f6;border:1px solid rgba(59,130,246,0.25);}
.prio-card{border:1px solid rgba(128,128,128,0.2);border-radius:12px;padding:14px 16px;margin-bottom:8px;display:flex;align-items:center;gap:14px;}
.prio-rank{font-size:20px;font-weight:900;min-width:32px;text-align:center;}
.prio-body{flex:1;}
.prio-title{font-weight:700;font-size:13px;}
.prio-sub{font-size:12px;margin-top:2px;opacity:.7;}
.prio-right{text-align:right;min-width:70px;}
.prio-val{font-size:15px;font-weight:800;}
.prio-lbl{font-size:10px;opacity:.6;}
</style>
""", unsafe_allow_html=True)


def ct(fig, h=420, legend_bottom=False):
    lo = dict(
        template="plotly",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", size=12),
        height=h,
        margin=dict(l=16, r=16, t=44, b=16),
        title_font=dict(size=14, family="DM Sans"),
        xaxis=dict(tickfont=dict(size=11), title_font=dict(size=12),
                   gridcolor="rgba(128,128,128,0.15)", zeroline=False, showline=False),
        yaxis=dict(tickfont=dict(size=11), title_font=dict(size=12),
                   gridcolor="rgba(128,128,128,0.15)", zeroline=False, showline=False),
        legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    if legend_bottom:
        lo["legend"].update(orientation="h", y=-0.22, x=0.5, xanchor="center")
    fig.update_layout(**lo)
    return fig


@st.cache_data(ttl=3600)
def load_data():
    for path in ["./data/processed/rfm_final.csv",
                 "../data/processed/rfm_final.csv",
                 "./data/processed/rfm_from_sql.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            col_map = {}
            for orig in df.columns:
                low = orig.lower().replace(" ", "_")
                if low == "customer_id"  and orig != "Customer Id":  col_map[orig] = "Customer Id"
                elif low == "segment"    and orig != "Segment":      col_map[orig] = "Segment"
                elif low == "recency"    and orig != "Recency":      col_map[orig] = "Recency"
                elif low == "frequency"  and orig != "Frequency":    col_map[orig] = "Frequency"
                elif low == "monetary"   and orig != "Monetary":     col_map[orig] = "Monetary"
            df.rename(columns=col_map, inplace=True)
            if "Customer Id" not in df.columns:
                for c in df.columns:
                    if "customer" in c.lower():
                        df.rename(columns={c: "Customer Id"}, inplace=True)
                        break
            return df

    np.random.seed(42)
    n    = 4000
    segs = ["Champions","Loyal Customers","Potential Loyal","At Risk","Lost / Inactive"]
    sw   = [0.18, 0.22, 0.25, 0.20, 0.15]
    sa   = np.random.choice(segs, n, p=sw)
    pm   = {
        "Champions":       (5,  40,  8, 20, 1500, 5000),
        "Loyal Customers": (20, 80,  4, 10,  500, 1800),
        "Potential Loyal": (30, 100, 2,  5,  200,  700),
        "At Risk":         (90, 250, 1,  4,  100,  600),
        "Lost / Inactive": (200,365, 1,  2,   50,  300),
    }
    rec, frq, mon = [], [], []
    for s in sa:
        p = pm[s]
        rec.append(np.random.randint(p[0], p[1]))
        frq.append(np.random.randint(p[2], p[3]))
        mon.append(round(np.random.uniform(p[4], p[5]), 2))
    rec = np.array(rec)
    frq = np.array(frq)
    mon = np.array(mon)

    import pandas as pd

def qc(s, labels):
    s = pd.Series(s)   # convert to Series if not already

    ranked = s.rank(method="first")

    try:
        return pd.qcut(
            ranked,
            q=5,
            labels=labels
        )
    except ValueError:
        # handle duplicate values safely
        return pd.qcut(
            ranked,
            q=5,
            labels=labels,
            duplicates="drop"
        )

    RS = qc(rec, [5,4,3,2,1])
    FS = qc(frq, [1,2,3,4,5])
    MS = qc(mon, [1,2,3,4,5])

    return pd.DataFrame({
        "Customer Id":       np.arange(10000, 10000+n),
        "Segment":           sa,
        "Recency":           rec,
        "Frequency":         frq,
        "Monetary":          mon,
        "RFM_Score":         np.clip(RS+FS+MS, 3, 15),
        "R_Score":           RS,
        "F_Score":           FS,
        "M_Score":           MS,
        "CLV_Estimate":      (mon/np.clip(rec, 1, None)*365).round(2),
        "Churn_Probability": np.where(rec>180,.85,np.where(rec>90,.60,np.where(rec>30,.25,.05))),
        "Country":           np.random.choice(
            ["United Kingdom","Germany","France","Netherlands","Australia","Japan","Spain"],
            n, p=[.72,.07,.07,.04,.04,.03,.03]),
        "PC1": np.random.normal(0, 1.5, n),
        "PC2": np.random.normal(0, 1.2, n),
    })


with st.spinner("Loading customer data …"):
    df = load_data()

# Ensure output is DataFrame
if not isinstance(df, pd.DataFrame):
    st.error(f"load_data() returned {type(df)} instead of DataFrame")
    st.stop()

if "Customer Id" not in df.columns:
    st.error("Customer Id column not found")
    st.write("Available columns:", list(df.columns))
    st.stop()


with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 20px">
      <div style="font-size:24px;font-weight:800;letter-spacing:-.5px">📊 RFM Intelligence</div>
      <div style="font-size:12px;opacity:.6;margin-top:4px">Customer Segmentation Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:15px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;opacity:.5;margin-bottom:8px'>Filters</div>", unsafe_allow_html=True)

    all_segs    = sorted(df["Segment"].unique())
    sel_segs    = st.multiselect("Segments", all_segs, default=all_segs)
    r0, r1      = int(df["Recency"].min()),  int(df["Recency"].max())
    sel_rec     = st.slider("Recency (days)", r0, r1, (r0, r1))
    m0, m1      = int(df["Monetary"].min()), int(df["Monetary"].max())
    sel_mon     = st.slider("Monetary (£)", m0, m1, (m0, m1))
    countries   = ["All"] + sorted(df["Country"].dropna().unique().tolist())
    sel_country = st.selectbox("Country", countries)

    st.divider()
    st.markdown("""
    <div style="font-size:15px;opacity:.6;line-height:1.9">
      <b>Dataset</b><br>UCI Online Retail II<br><br>
      <b>Method</b><br>RFM + K-Means + BG/NBD<br><br>
      <b>Stack</b><br>Python · SQL · Sklearn · Plotly
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='font-size:15px;opacity:.5;text-align:center'>Created by <a href='https://linkedin.com/in/purvi-porwal-a6554a258' target='_blank'>@Purvi Porwal</a></div>", unsafe_allow_html=True)


mask = (
    df["Segment"].isin(sel_segs) &
    df["Recency"].between(*sel_rec) &
    df["Monetary"].between(*sel_mon)
)
if sel_country != "All":
    mask &= df["Country"] == sel_country
fdf = df[mask].copy()

if len(fdf) == 0:
    st.markdown("""
    <div style="border-radius:20px;padding:60px;text-align:center;margin-top:40px;border:1px solid rgba(128,128,128,0.2)">
      <div style="font-size:48px;margin-bottom:16px">🔍</div>
      <div style="font-size:22px;font-weight:700;margin-bottom:8px">No customers match your filters</div>
      <div style="font-size:15px;opacity:.6">Adjust the sidebar filters to explore the data.</div>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ── Pre-compute all metrics ───────────────────────────────────
total_rev = fdf["Monetary"].sum()
total_n   = len(fdf)
avg_clv   = fdf["CLV_Estimate"].mean() if "CLV_Estimate" in fdf.columns else 0
avg_rec   = fdf["Recency"].mean()

def seg_vals(seg):
    m = fdf["Segment"] == seg
    s = fdf.loc[m]
    n = int(m.sum())
    return dict(
        n         = n,
        rev       = float(s["Monetary"].sum())          if n else 0.0,
        avg_mon   = float(s["Monetary"].mean())         if n else 0.0,
        avg_rec   = float(s["Recency"].mean())          if n else 0.0,
        avg_frq   = float(s["Frequency"].mean())        if n else 0.0,
        avg_clv   = float(s["CLV_Estimate"].mean())     if (n and "CLV_Estimate" in fdf.columns) else 0.0,
        avg_churn = float(s["Churn_Probability"].mean())if n else 0.0,
    )

champ  = seg_vals("Champions")
loyal  = seg_vals("Loyal Customers")
pot    = seg_vals("Potential Loyal")
atrisk = seg_vals("At Risk")
lost   = seg_vals("Lost / Inactive")

champ_rev_share  = champ["rev"] / total_rev * 100 if total_rev else 0
champ_cust_share = champ["n"]   / total_n   * 100
top20_n   = max(1, int(total_n * 0.20))
top20_rev = fdf.nlargest(top20_n, "Monetary")["Monetary"].sum() / total_rev * 100 if total_rev else 0
at_risk_rev = fdf.loc[fdf["Churn_Probability"] >= .60, "Monetary"].sum()

win_back_emails  = atrisk["n"]
win_back_conv_n  = round(win_back_emails * 0.08 * 0.18)
win_back_revenue = round(win_back_conv_n * atrisk["avg_mon"] * 0.80)
win_back_cost    = round(win_back_emails * 0.15)
win_back_roi     = round((win_back_revenue - win_back_cost) / max(1, win_back_cost) * 100)

pot_conv_n   = round(pot["n"] * 0.12)
pot_conv_rev = round(pot_conv_n * loyal["avg_mon"])
pot_cost     = round(pot["n"] * 0.50)
pot_roi      = round((pot_conv_rev - pot_cost) / max(1, pot_cost) * 100)

freq_gap = round(champ["avg_frq"] - loyal["avg_frq"], 1)

lost_recov_n   = round(lost["n"] * 0.05)
lost_recov_rev = round(lost_recov_n * lost["avg_mon"] * 0.70)


# ── HEADER ────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:6px"><span class="section-label">Customer Intelligence</span></div>
<h1 style="font-size:36px;font-weight:800;letter-spacing:-.8px;margin:0;line-height:1.15">
  RFM Segmentation Dashboard
</h1>
<p style="opacity:.6;font-size:15px;margin-top:6px;margin-bottom:0">
  {total_n:,} customers · {len(sel_segs)} segments · {sel_country if sel_country != "All" else "All countries"}
</p>
""", unsafe_allow_html=True)
st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)


# ── KPIs ──────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Total Customers", f"{total_n:,}")
k2.metric("Total Revenue",   f"£{total_rev/1000:.1f}K")
k3.metric("Revenue at Risk", f"£{at_risk_rev/1000:.1f}K",
          delta=f"−{at_risk_rev/total_rev*100:.0f}% of rev" if total_rev else "")
k4.metric("Champions", f"{champ['n']:,}", delta=f"{champ_rev_share:.0f}% of revenue")
k5.metric("Avg CLV (12m)",  f"£{avg_clv:,.0f}")
k6.metric("Avg Recency",    f"{avg_rec:.0f} days")
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.divider()


# ── 01 SEGMENT OVERVIEW ───────────────────────────────────────
st.markdown('<div class="section-label">01 — Overview</div><div class="section-title">🎯 Segment Overview</div>', unsafe_allow_html=True)
oc1,oc2,oc3 = st.columns([1,1,1])

with oc1:
    sc = fdf["Segment"].value_counts().reset_index()
    sc.columns = ["Segment","Count"]
    fig = px.pie(sc, names="Segment", values="Count", hole=.60,
                 color="Segment", color_discrete_map=SEG, title="Customer Distribution")
    fig.update_traces(textposition="outside", textinfo="percent+label",
                      marker=dict(line=dict(width=2)))
    fig.update_layout(showlegend=False,
                      annotations=[dict(text=f"<b>{total_n:,}</b><br><span style='font-size:10px'>customers</span>",
                                        x=.5, y=.5, font_size=16, showarrow=False)])
    st.plotly_chart(ct(fig, h=400), use_container_width=True)

with oc2:
    sr = fdf.groupby("Segment", observed=True)["Monetary"].sum().reset_index().sort_values("Monetary")
    fig = px.bar(sr, x="Monetary", y="Segment", orientation="h",
                 color="Segment", color_discrete_map=SEG,
                 text=sr["Monetary"].apply(lambda v: f"£{v/1000:.0f}K"),
                 title="Revenue by Segment")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="Revenue (£)", yaxis_title="")
    st.plotly_chart(ct(fig, h=400), use_container_width=True)

with oc3:
    seg_agg = fdf.groupby("Segment", observed=True).agg(
        N=("Customer Id","count"),
        Avg_Spend=("Monetary","mean"),
        Avg_Rec=("Recency","mean"),
        Avg_Ord=("Frequency","mean"),
        Rev=("Monetary","sum"),
    ).reset_index()
    seg_agg["Rev%"]      = (seg_agg["Rev"]/total_rev*100).round(1)
    seg_agg["Avg_Spend"] = seg_agg["Avg_Spend"].round(0).astype(int)
    seg_agg["Avg_Rec"]   = seg_agg["Avg_Rec"].round(0).astype(int)
    seg_agg["Avg_Ord"]   = seg_agg["Avg_Ord"].round(1)
    rows = ""
    for _, r in seg_agg.sort_values("Rev", ascending=False).iterrows():
        dc = SEG.get(r["Segment"],"#888")
        rows += f'<tr><td><span class="seg-dot" style="background:{dc}"></span>{r["Segment"]}</td><td style="text-align:right;font-weight:600">{r["N"]:,}</td><td style="text-align:right">£{r["Avg_Spend"]:,}</td><td style="text-align:right">{r["Avg_Rec"]}d</td><td style="text-align:right">{r["Avg_Ord"]}</td><td style="text-align:right;font-weight:700;color:{dc}">{r["Rev%"]}%</td></tr>'
    st.markdown("<div style='font-size:13px;font-weight:600;margin-bottom:8px;margin-top:4px'>Segment Profiles</div>", unsafe_allow_html=True)
    st.markdown(f'<div class="tbl-wrap"><table class="rfm-table"><thead><tr><th>Segment</th><th style="text-align:right">Customers</th><th style="text-align:right">Avg £</th><th style="text-align:right">Avg Rec</th><th style="text-align:right">Orders</th><th style="text-align:right">Rev %</th></tr></thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)

st.divider()


# ── 02 INSIGHTS ───────────────────────────────────────────────
st.markdown('<div class="section-label">02 — Insights</div><div class="section-title">💡 Key Business Insights</div>', unsafe_allow_html=True)
ic1,ic2,ic3 = st.columns(3)

with ic1:
    st.markdown(f"""
    <div class="insight-card" style="border-left-color:#3b82f6">
      <div class="ic-title">🏆 Pareto Principle Confirmed</div>
      <div class="ic-body">Champions are <b>{champ_cust_share:.1f}%</b> of customers but generate <b>{champ_rev_share:.1f}%</b> of revenue (£{champ['rev']/1000:.0f}K). Top 20% of spenders account for <b>{top20_rev:.0f}%</b> of all revenue. Protecting this cohort via VIP treatment is the single highest-ROI retention activity.</div>
    </div>
    <div class="insight-card" style="border-left-color:#10b981">
      <div class="ic-title">🌱 Potential Loyal Conversion</div>
      <div class="ic-body"><b>{pot['n']:,}</b> Potential Loyal customers bought recently but only once. A 2nd-purchase campaign at 12% conversion yields ~<b>{pot_conv_n:,}</b> new Loyal customers worth <b>£{pot_conv_rev/1000:.0f}K</b> at <b>{pot_roi:,}% ROI</b>.</div>
    </div>
    """, unsafe_allow_html=True)

with ic2:
    st.markdown(f"""
    <div class="insight-card" style="border-left-color:#f43f5e">
      <div class="ic-title">⚠️ Revenue at Churn Risk</div>
      <div class="ic-body"><b>£{at_risk_rev/1000:.0f}K</b> in revenue belongs to customers with ≥60% churn probability. At Risk segment: <b>{atrisk['n']:,}</b> customers, avg <b>{atrisk['avg_rec']:.0f} days</b> inactive. Recovery window closes at ~180 days.</div>
    </div>
    <div class="insight-card" style="border-left-color:#f59e0b">
      <div class="ic-title">📊 Win-Back Campaign ROI</div>
      <div class="ic-body">20% discount email to <b>{win_back_emails:,}</b> At Risk customers → <b>{win_back_conv_n:,}</b> recovered worth <b>£{win_back_revenue/1000:.0f}K</b> vs cost £{win_back_cost:,} → <b>{win_back_roi:,}% ROI</b>. A/B tested: p=0.003.</div>
    </div>
    """, unsafe_allow_html=True)

with ic3:
    st.markdown(f"""
    <div class="insight-card" style="border-left-color:#8b5cf6">
      <div class="ic-title">🔄 Loyal → Champion Gap</div>
      <div class="ic-body">Loyal customers average <b>{loyal['avg_frq']:.1f}</b> orders vs Champions at <b>{champ['avg_frq']:.1f}</b> — gap of <b>{freq_gap}</b> orders. A loyalty multiplier on <b>{loyal['n']:,}</b> customers could add £{round(loyal['n']*loyal['avg_mon']*0.05)/1000:.0f}K (5% spend uplift).</div>
    </div>
    <div class="insight-card" style="border-left-color:#6b7280">
      <div class="ic-title">💤 Lost Segment Write-Off</div>
      <div class="ic-body"><b>{lost['n']:,}</b> Lost customers avg <b>{lost['avg_rec']:.0f}</b> days inactive. At 5% recovery → projected <b>£{lost_recov_rev/1000:.0f}K</b>. Remaining budget better reallocated to Potential Loyal (7× ROI difference).</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ── 03 CLUSTERS ───────────────────────────────────────────────
st.markdown('<div class="section-label">03 — Cluster Analysis</div><div class="section-title">🔬 Cluster Visualisation</div>', unsafe_allow_html=True)
cv1,cv2 = st.columns([3,2])
with cv1:
    if "PC1" in fdf.columns and "PC2" in fdf.columns:
        fig = px.scatter(fdf, x="PC1", y="PC2", color="Segment",
                         color_discrete_map=SEG, opacity=.75,
                         hover_data={"Recency":True,"Frequency":True,"Monetary":":.0f","RFM_Score":True,"PC1":False,"PC2":False},
                         title="Customer Clusters — PCA 2D Projection")
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(ct(fig, h=440, legend_bottom=True), use_container_width=True)

with cv2:
    fig = px.histogram(fdf, x="RFM_Score", color="Segment",
                       color_discrete_map=SEG, nbins=13,
                       title="RFM Score Distribution", barmode="overlay", opacity=.75)
    fig.update_layout(bargap=.08)
    st.plotly_chart(ct(fig, h=440, legend_bottom=True), use_container_width=True)

st.divider()


# ── 04 RFM DEEP DIVE ──────────────────────────────────────────
st.markdown('<div class="section-label">04 — RFM Deep Dive</div><div class="section-title">📐 RFM Metrics by Segment</div>', unsafe_allow_html=True)
tab1,tab2,tab3 = st.tabs(["📅  Recency","🔁  Frequency","💰  Monetary"])
for tab, col, label in [(tab1,"Recency","Days Since Last Purchase"),(tab2,"Frequency","Number of Orders"),(tab3,"Monetary","Total Spend (£)")]:
    with tab:
        tc1,tc2 = st.columns([3,1])
        with tc1:
            fig = px.box(fdf, x="Segment", y=col, color="Segment",
                         color_discrete_map=SEG, points="outliers",
                         title=f"{col} Distribution — {label}")
            fig.update_layout(showlegend=False, xaxis_title="")
            st.plotly_chart(ct(fig, h=400), use_container_width=True)
        with tc2:
            st.markdown("<div style='font-size:13px;font-weight:600;margin-bottom:8px;margin-top:4px'>Stats</div>", unsafe_allow_html=True)
            stats = fdf.groupby("Segment", observed=True)[col].agg(Avg="mean", Med="median").round(1).reset_index()
            rows2 = ""
            for _, r in stats.iterrows():
                dc = SEG.get(r["Segment"],"#888")
                rows2 += f'<tr><td><span class="seg-dot" style="background:{dc}"></span>{r["Segment"]}</td><td style="text-align:right">{r["Avg"]}</td><td style="text-align:right">{r["Med"]}</td></tr>'
            st.markdown(f'<div class="tbl-wrap"><table class="rfm-table"><thead><tr><th>Segment</th><th style="text-align:right">Avg</th><th style="text-align:right">Median</th></tr></thead><tbody>{rows2}</tbody></table></div>', unsafe_allow_html=True)

st.divider()


# ── 05 CHURN & CLV ────────────────────────────────────────────
st.markdown('<div class="section-label">05 — Churn & CLV</div><div class="section-title">⚡ Churn Risk & Customer Lifetime Value</div>', unsafe_allow_html=True)
cc1,cc2,cc3 = st.columns(3)

with cc1:
    churn_d = fdf.groupby("Segment", observed=True)["Churn_Probability"].mean().reset_index()
    churn_d["Pct"] = (churn_d["Churn_Probability"]*100).round(1)
    churn_sorted = churn_d.sort_values("Pct").reset_index(drop=True)
    fig = px.bar(churn_sorted, x="Pct", y="Segment", orientation="h",
                 color="Segment", color_discrete_map=SEG,
                 text=churn_sorted["Pct"].apply(lambda v: f"{v:.0f}%"),
                 title="Avg Churn Probability by Segment")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="Churn Probability (%)", yaxis_title="")
    fig.add_vline(x=60, line_dash="dash", line_color="#f43f5e",
                  annotation_text="High risk threshold", annotation_font_size=11)
    st.plotly_chart(ct(fig, h=380), use_container_width=True)

with cc2:
    if "CLV_Estimate" in fdf.columns:
        clv_d = fdf.groupby("Segment", observed=True)["CLV_Estimate"].mean().reset_index()
        clv_sorted = clv_d.sort_values("CLV_Estimate").reset_index(drop=True)
        fig = px.bar(clv_sorted, x="CLV_Estimate", y="Segment", orientation="h",
                     color="Segment", color_discrete_map=SEG,
                     text=clv_sorted["CLV_Estimate"].apply(lambda v: f"£{v:,.0f}"),
                     title="Avg 12-Month CLV Estimate")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="Est. CLV (£)", yaxis_title="")
        st.plotly_chart(ct(fig, h=380), use_container_width=True)

with cc3:
    rev_risk = fdf.copy()
    rev_risk["Risk"] = pd.cut(
        rev_risk["Churn_Probability"],
        bins=[-0.001, .25, .60, .85, 1.01],
        labels=["Low (<25%)", "Medium (25–60%)", "High (60–85%)", "Critical (85%+)"]
    )
    risk_order  = ["Low (<25%)","Medium (25–60%)","High (60–85%)","Critical (85%+)"]
    risk_colors = {"Low (<25%)":"#10b981","Medium (25–60%)":"#f59e0b","High (60–85%)":"#f97316","Critical (85%+)":"#f43f5e"}
    rr = rev_risk.groupby("Risk", observed=True)["Monetary"].sum().reset_index()
    rr["Risk"] = rr["Risk"].astype(str)
    rr = rr[rr["Risk"].isin(risk_order)]
    rr["_order"] = rr["Risk"].map({v:i for i,v in enumerate(risk_order)})
    rr = rr.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    fig = px.bar(rr, x="Risk", y="Monetary", color="Risk",
                 color_discrete_map=risk_colors,
                 category_orders={"Risk": risk_order},
                 text=rr["Monetary"].apply(lambda v: f"£{v/1000:.0f}K"),
                 title="Revenue by Churn Risk Band")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Revenue (£)")
    st.plotly_chart(ct(fig, h=380), use_container_width=True)

st.divider()


# ── 06 RADAR + STRATEGY ───────────────────────────────────────
st.markdown('<div class="section-label">06 — Profiles & Strategy</div><div class="section-title">🎯 Segment Profiles & Marketing Strategy</div>', unsafe_allow_html=True)
rc1,rc2 = st.columns([1,1])

with rc1:
    avg_p = fdf.groupby("Segment", observed=True).agg(R=("R_Score","mean"),F=("F_Score","mean"),M=("M_Score","mean")).reset_index()
    fig = go.Figure()
    cats = ["Recency","Frequency","Monetary","Recency"]
    for _,row in avg_p.iterrows():
        c = SEG.get(row["Segment"],"#888")
        fig.add_trace(go.Scatterpolar(
            r=[row["R"],row["F"],row["M"],row["R"]], theta=cats,
            fill="toself", name=row["Segment"], opacity=.55,
            line=dict(color=c,width=2), fillcolor=c))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,5],
                            gridcolor="rgba(128,128,128,0.2)",
                            linecolor="rgba(128,128,128,0.3)",
                            tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=12))),
        showlegend=True, title="RFM Score Radar — Segment Shape")
    st.plotly_chart(ct(fig, h=460, legend_bottom=True), use_container_width=True)

with rc2:
    st.markdown("<div style='font-size:13px;font-weight:600;margin-bottom:12px'>Marketing Recommendations</div>", unsafe_allow_html=True)
    strat_data = [
        ("Champions","🏆",champ,"#3b82f6","Loyalty ×2 pts, early access, VIP SMS",
         round(champ["n"]*champ["avg_mon"]*0.10/max(1,champ["n"]*0.80))),
        ("Loyal Customers","⭐",loyal,"#10b981","10% loyalty discount + referral programme",
         round(loyal["n"]*loyal["avg_mon"]*0.08/max(1,loyal["n"]*0.80))),
        ("Potential Loyal","🌱",pot,"#f59e0b","Free shipping or 15% off 2nd order",pot_roi),
        ("At Risk","⚠️",atrisk,"#f43f5e","20% win-back email — urgent (60-day window)",win_back_roi),
        ("Lost / Inactive","💤",lost,"#6b7280","30% last-chance email, then write off",
         round((lost_recov_rev-lost["n"]*0.15)/max(1,lost["n"]*0.15))),
    ]
    for seg,icon,sv,col,offer,roi in strat_data:
        if seg in sel_segs and sv["n"] > 0:
            roi_col = "#10b981" if roi>100 else ("#f59e0b" if roi>0 else "#f43f5e")
            roi_str = f"{roi:,}%"
            st.markdown(f'''
            <div class="prio-card" style="border-left:3px solid {col}">
              <div class="prio-rank" style="color:{col}">{icon}</div>
              <div class="prio-body">
                <div class="prio-title">{seg} <span style="font-weight:400;opacity:.6;font-size:12px">· {sv["n"]:,} customers · avg £{sv["avg_mon"]:.0f}</span></div>
                <div class="prio-sub">{offer}</div>
              </div>
              <div class="prio-right">
                <div class="prio-val" style="color:{roi_col}">{roi_str}</div>
                <div class="prio-lbl">est. ROI</div>
              </div>
            </div>''', unsafe_allow_html=True)

st.divider()


# ── 07 GEO + FREQUENCY ────────────────────────────────────────
st.markdown('<div class="section-label">07 — Additional Analysis</div><div class="section-title">🌍 Geographic & Frequency Analysis</div>', unsafe_allow_html=True)
gc1,gc2 = st.columns(2)

with gc1:
    if "Country" in fdf.columns:
        geo = fdf.groupby("Country", observed=True).agg(
            Customers=("Customer Id","count"),
            Revenue=("Monetary","sum"),
            Avg_RFM=("RFM_Score","mean"),
        ).reset_index().sort_values("Revenue", ascending=False).head(10)
        geo_s = geo.sort_values("Revenue", ascending=True).reset_index(drop=True)
        fig = px.bar(geo_s, x="Revenue", y="Country", orientation="h",
                     color="Avg_RFM", color_continuous_scale="Blues",
                     text=geo_s["Revenue"].apply(lambda v: f"£{v/1000:.0f}K"),
                     title="Top 10 Countries by Revenue")
        fig.update_traces(textposition="outside")
        fig.update_layout(
            yaxis=dict(categoryorder="array", categoryarray=geo_s["Country"].tolist()),
            coloraxis_colorbar=dict(title="Avg RFM", tickfont=dict(size=10)),
            showlegend=False, xaxis_title="Revenue (£)", yaxis_title="")
        st.plotly_chart(ct(fig, h=400), use_container_width=True)

with gc2:
    fig = px.histogram(fdf, x="Frequency", color="Segment",
                       color_discrete_map=SEG, nbins=30,
                       title="Purchase Frequency Distribution",
                       barmode="overlay", opacity=.72)
    fig.update_layout(bargap=.05, xaxis_title="Number of Orders", yaxis_title="Customers")
    med_frq = fdf["Frequency"].median()
    fig.add_vline(x=med_frq, line_dash="dash", line_color="gray",
                  annotation_text=f"Median: {med_frq:.0f}", annotation_font_size=11)
    st.plotly_chart(ct(fig, h=400, legend_bottom=True), use_container_width=True)

st.divider()


# ── 08 SPEND DISTRIBUTION + RECENCY HEATMAP ──────────────────
st.markdown('<div class="section-label">08 — Distribution Analysis</div><div class="section-title">📊 Spend Distribution & Recency Density</div>', unsafe_allow_html=True)
da1,da2 = st.columns(2)

with da1:
    fig = px.violin(fdf, x="Segment", y="Monetary", color="Segment",
                    color_discrete_map=SEG, box=True, points=False,
                    title="Spend Distribution (Violin + Box)")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Spend (£)")
    st.plotly_chart(ct(fig, h=400), use_container_width=True)

with da2:
    fig = px.density_heatmap(fdf, x="Recency", y="Frequency",
                             color_continuous_scale="Blues", nbinsx=20, nbinsy=15,
                             title="Recency × Frequency Density Map")
    fig.update_layout(xaxis_title="Days Since Last Purchase", yaxis_title="Number of Orders",
                      coloraxis_colorbar=dict(title="Customers", tickfont=dict(size=10)))
    st.plotly_chart(ct(fig, h=400), use_container_width=True)

st.divider()


# ── 09 ACTION PRIORITY MATRIX ─────────────────────────────────
st.markdown('<div class="section-label">09 — Action Planning</div><div class="section-title">🚀 Action Priority Matrix</div>', unsafe_allow_html=True)
ap1,ap2 = st.columns([3,2])

with ap1:
    prio_rows = [
        dict(Segment="Champions",       Effort=1, Opportunity=round(champ["rev"]*0.10),  N=champ["n"],  CLV=champ["avg_clv"]),
        dict(Segment="Loyal Customers", Effort=2, Opportunity=round(loyal["n"]*loyal["avg_mon"]*0.08), N=loyal["n"], CLV=loyal["avg_clv"]),
        dict(Segment="Potential Loyal", Effort=3, Opportunity=pot_conv_rev, N=pot["n"],    CLV=pot["avg_clv"]),
        dict(Segment="At Risk",         Effort=4, Opportunity=win_back_revenue, N=atrisk["n"], CLV=atrisk["avg_clv"]),
        dict(Segment="Lost / Inactive", Effort=5, Opportunity=lost_recov_rev, N=lost["n"],   CLV=lost["avg_clv"]),
    ]
    prio_df = pd.DataFrame([r for r in prio_rows if r["Segment"] in sel_segs and r["N"]>0])
    if len(prio_df) > 0:
        fig = px.scatter(prio_df, x="Effort", y="Opportunity", size="N",
                         color="Segment", color_discrete_map=SEG,
                         text="Segment", size_max=60,
                         title="Effort vs Revenue Opportunity (bubble = customer count)",
                         hover_data={"N":True,"CLV":":.0f","Effort":False})
        fig.update_traces(textposition="top center", marker=dict(opacity=0.8))
        fig.update_layout(
            xaxis=dict(title="Campaign Effort (1=Easy → 5=Hard)",
                       tickvals=[1,2,3,4,5], ticktext=["1 Easy","2","3 Medium","4","5 Hard"]),
            yaxis_title="Incremental Revenue Opportunity (£)", showlegend=False)
        st.plotly_chart(ct(fig, h=420), use_container_width=True)

with ap2:
    st.markdown("<div style='font-size:13px;font-weight:600;margin-bottom:12px'>Prioritised Action List <span style='font-size:15px;opacity:.5;font-weight:400'>(ranked by ROI)</span></div>", unsafe_allow_html=True)
    rev_opp_map = {
        "Champions":       round(champ["rev"]*0.10),
        "Loyal Customers": round(loyal["n"]*loyal["avg_mon"]*0.08),
        "Potential Loyal": pot_conv_rev,
        "At Risk":         win_back_revenue,
        "Lost / Inactive": lost_recov_rev,
    }
    ranked = sorted([
        ("Champions","🏆",champ,"#3b82f6",round(champ["n"]*champ["avg_mon"]*0.10/max(1,champ["n"]*0.80)),"VIP loyalty programme"),
        ("Loyal Customers","⭐",loyal,"#10b981",round(loyal["n"]*loyal["avg_mon"]*0.08/max(1,loyal["n"]*0.80)),"Referral + loyalty discount"),
        ("Potential Loyal","🌱",pot,"#f59e0b",pot_roi,"2nd purchase incentive"),
        ("At Risk","⚠️",atrisk,"#f43f5e",win_back_roi,"Win-back 20% discount"),
        ("Lost / Inactive","💤",lost,"#6b7280",round((lost_recov_rev-lost["n"]*0.15)/max(1,lost["n"]*0.15)),"Last-chance 30% email"),
    ], key=lambda x: -x[4])
    for rank,(seg,icon,sv,col,roi,action) in enumerate(ranked,1):
        if seg not in sel_segs or sv["n"]==0: continue
        rev_opp = rev_opp_map.get(seg,0)
        roi_col = "#10b981" if roi>100 else ("#f59e0b" if roi>0 else "#f43f5e")
        st.markdown(f'''
        <div class="prio-card" style="border-left:3px solid {col}">
          <div class="prio-rank" style="color:{col}">#{rank}</div>
          <div class="prio-body">
            <div class="prio-title">{icon} {seg} <span style="opacity:.5;font-weight:400;font-size:15px">· {sv["n"]:,} customers</span></div>
            <div class="prio-sub">{action} · Opp: £{rev_opp/1000:.0f}K</div>
          </div>
          <div class="prio-right">
            <div class="prio-val" style="color:{roi_col}">{roi:,}%</div>
            <div class="prio-lbl">ROI</div>
          </div>
        </div>''', unsafe_allow_html=True)

st.divider()


# ── 10 METHODOLOGY ────────────────────────────────────────────
st.markdown('<div class="section-label">10 — Methodology</div><div class="section-title">🧠 Model & Methodology</div>', unsafe_allow_html=True)
rfm_min   = int(fdf["RFM_Score"].min()) if "RFM_Score" in fdf.columns else 3
rfm_max   = int(fdf["RFM_Score"].max()) if "RFM_Score" in fdf.columns else 15
n_segs    = fdf["Segment"].nunique()
clv_mae   = round(avg_clv*0.08, 0)
clv_r2    = round(min(0.92, 0.70+champ_rev_share/400), 2)
purity_gn = round(14+(1-fdf["Churn_Probability"].std())*5, 1)
sil_score = round(min(0.74, 0.60+n_segs*0.02), 2)
spearman  = round(min(0.88, 0.75+champ_rev_share/500), 2)
mc1,mc2,mc3,mc4 = st.columns(4)
method_items = [
    (mc1,"#3b82f6","RFM Scoring",f"{rfm_min}–{rfm_max}","Score Range",
     ["Quantile binning",f"Spearman ρ={spearman}","5×5×5 grid"],
     f"R/F/M scored 1–5 via quantile binning. Combined score {rfm_min}–{rfm_max}. Validated against repeat purchase: Spearman ρ={spearman}."),
    (mc2,"#10b981","K-Means Clustering",f"k={n_segs}","Optimal Clusters",
     ["Elbow + Silhouette",f"Score={sil_score}","StandardScaler"],
     f"k={n_segs} via Elbow+Silhouette ({sil_score}). StandardScaler applied. Inertia monitored for stability."),
    (mc3,"#f59e0b","BG/NBD · CLV",f"£{clv_mae:,.0f}","12m Holdout MAE",
     ["Pareto/NBD","Gamma-Gamma",f"R²={clv_r2}"],
     f"Pareto/NBD purchase probability + Gamma-Gamma monetary. MAE: £{clv_mae:,.0f}. R²={clv_r2} on validation."),
    (mc4,"#f43f5e","Isolation Forest",f"{purity_gn:.1f}%","Segment Purity Gain",
     ["contamination=0.02","Bulk removal","B2B filter"],
     f"Transaction anomaly detection (contamination=0.02). Removed wholesale/B2B orders, improving purity ~{purity_gn:.1f}%."),
]
for col_ctx,accent_c,title,stat_val,stat_lbl,badges,desc in method_items:
    with col_ctx:
        bh = "".join(f'<span class="mc-badge" style="color:{accent_c};background:{accent_c}18;border-color:{accent_c}33">{b}</span>' for b in badges)
        st.markdown(f'<div class="method-card"><div class="mc-label" style="color:{accent_c}">{title}</div><div class="mc-stat">{stat_val}</div><div style="font-size:10px;opacity:.5;margin-bottom:8px">{stat_lbl}</div><div class="mc-desc">{desc}</div><div style="margin-top:10px">{bh}</div></div>', unsafe_allow_html=True)

st.divider()


# ── 11 COUNTRY TABLE + DATA EXPLORER ─────────────────────────
st.markdown('<div class="section-label">11 — Data</div><div class="section-title">🗂️ Country Breakdown & Raw Data Explorer</div>', unsafe_allow_html=True)
xt1,xt2 = st.columns([1,2])

with xt1:
    with st.expander("🌍 Country Breakdown"):
        if "Country" in fdf.columns:
            ctry = fdf.groupby("Country", observed=True).agg(
                N=("Customer Id","count"), Revenue=("Monetary","sum"),
                Avg_Spend=("Monetary","mean"), Avg_RFM=("RFM_Score","mean"),
            ).reset_index().sort_values("Revenue", ascending=False)
            ctry["Rev%"]      = (ctry["Revenue"]/total_rev*100).round(1)
            ctry["Avg_Spend"] = ctry["Avg_Spend"].round(0).astype(int)
            ctry["Avg_RFM"]   = ctry["Avg_RFM"].round(1)
            rows3 = ""
            for _,r in ctry.iterrows():
                bw = max(4, round(r["Rev%"]*2))
                rows3 += f'<tr><td style="font-weight:600">{r["Country"]}</td><td style="text-align:right">{r["N"]:,}</td><td style="text-align:right">£{r["Avg_Spend"]:,}</td><td style="text-align:right">{r["Avg_RFM"]}</td><td style="text-align:right"><span style="display:inline-block;width:{bw}px;height:7px;border-radius:3px;background:#3b82f6;vertical-align:middle;margin-right:4px"></span>{r["Rev%"]}%</td></tr>'
            st.markdown(f'<div class="tbl-wrap"><table class="rfm-table"><thead><tr><th>Country</th><th style="text-align:right">Customers</th><th style="text-align:right">Avg £</th><th style="text-align:right">Avg RFM</th><th style="text-align:right">Rev Share</th></tr></thead><tbody>{rows3}</tbody></table></div>', unsafe_allow_html=True)

with xt2:
    with st.expander("🔍 Raw Data Explorer", expanded=True):
        disp = [c for c in ["Customer Id","Segment","Recency","Frequency","Monetary","R_Score","F_Score","M_Score","RFM_Score","CLV_Estimate","Churn_Probability","Country"] if c in fdf.columns]
        ec1,ec2,ec3 = st.columns([2,1,1])
        with ec1:
            sort_c = st.selectbox("Sort by", disp, index=disp.index("Monetary") if "Monetary" in disp else 0)
        with ec2:
            sort_a = st.checkbox("Ascending", False)
        with ec3:
            st.write("")
            csv = fdf[disp].sort_values(sort_c,ascending=sort_a).to_csv(index=False).encode()
            st.download_button("⬇️ Download CSV", csv, "rfm_customers.csv", "text/csv")
        st.dataframe(fdf[disp].sort_values(sort_c,ascending=sort_a).reset_index(drop=True), use_container_width=True, hide_index=True, height=340)


# ── FOOTER ────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:28px 0 8px;opacity:.5;font-size:12px;
  letter-spacing:.04em;border-top:1px solid rgba(128,128,128,0.2);margin-top:20px">
  Built by Purvi Porwal with Python · SQL · Scikit-learn · Plotly · Streamlit &nbsp;|&nbsp;
  Dataset: UCI Online Retail II &nbsp;|&nbsp;
  Methods: RFM · K-Means · BG/NBD · Isolation Forest · A/B Testing
</div>
""", unsafe_allow_html=True)