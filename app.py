"""
Pump Early Warning System - Evaluasi Komparatif Dashboard
XGBoost vs Lag-LLaMA | Altcoin Mid-Cap
Skripsi: Mohamad Galih Prasetyo Adi - USB YPKP Bandung, 2026
Run:     streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import average_precision_score, precision_recall_curve
import json, os
from collections import defaultdict

st.set_page_config(
    page_title="Pump EWS Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === CUSTOM CSS ===
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
:root {
    --bg-deep:#050810;--bg-panel:#0c1021;--bg-card:#111631;--bg-hover:#161d3f;
    --accent-1:#6366f1;--accent-2:#22d3ee;--accent-3:#10b981;--accent-4:#f59e0b;--accent-5:#ef4444;
    --text-1:#f1f5f9;--text-2:#94a3b8;--text-3:#475569;--border:#1e2748;
    --glow:rgba(99,102,241,0.15);
}
.stApp { background: var(--bg-deep); font-family:'Outfit',sans-serif; }
.stApp > header { background:transparent !important; }
[data-testid="stSidebar"] { background:var(--bg-panel); border-right:1px solid var(--border); }
section[data-testid="stSidebar"] .stRadio > label { display:none; }
.m-card {
    background:linear-gradient(145deg,var(--bg-card),var(--bg-panel));
    border:1px solid var(--border); border-radius:16px; padding:22px 26px;
    position:relative; overflow:hidden; transition:all 0.25s ease;
}
.m-card:hover { border-color:var(--accent-1); box-shadow:0 0 30px var(--glow); transform:translateY(-2px); }
.m-card::before {
    content:''; position:absolute; top:0;left:0;right:0; height:3px;
    background:linear-gradient(90deg,var(--accent-1),var(--accent-2)); border-radius:16px 16px 0 0;
}
.m-label { font-family:'Outfit';font-size:11px;font-weight:600;color:var(--text-2);
           text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px; }
.m-value { font-family:'JetBrains Mono';font-size:30px;font-weight:700;color:var(--text-1);line-height:1.1; }
.m-sub { font-family:'JetBrains Mono';font-size:12px;color:var(--text-2);margin-top:6px; }
.m-delta-up { color:#10b981;font-family:'JetBrains Mono';font-size:13px;margin-top:4px; }
.m-delta-down { color:#ef4444;font-family:'JetBrains Mono';font-size:13px;margin-top:4px; }
.sec-h {
    font-family:'Outfit';font-size:11px;font-weight:700;color:var(--accent-2);
    text-transform:uppercase;letter-spacing:0.12em;padding-bottom:10px;
    border-bottom:1px solid var(--border);margin:28px 0 18px 0;
}
.insight-box {
    background:linear-gradient(135deg,rgba(99,102,241,0.08),rgba(34,211,238,0.05));
    border:1px solid rgba(99,102,241,0.25);border-radius:12px;padding:18px 22px;
    margin:12px 0;font-size:14px;color:var(--text-1);line-height:1.6;
}
.insight-box strong { color:var(--accent-2); }
.insight-box em { color:var(--accent-4);font-style:normal; }
.stTabs [data-baseweb="tab-list"] { gap:4px;background:var(--bg-panel);border-radius:10px;padding:4px;border:1px solid var(--border); }
.stTabs [data-baseweb="tab"] { border-radius:8px;font-family:'Outfit';font-weight:500;font-size:13px; }
.stTabs [aria-selected="true"] { background:var(--accent-1) !important;color:white !important; }
.sep { height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent);margin:20px 0; }
.page-title { font-family:'Outfit';font-size:36px;font-weight:800;
              background:linear-gradient(135deg,#f1f5f9,#94a3b8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px; }
.page-subtitle { font-family:'Outfit';font-size:15px;font-weight:400;color:var(--text-2);margin-bottom:16px; }
footer { visibility:hidden; }
#MainMenu { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# === DATA ===
DATA_DIR = "data/"

@st.cache_data(show_spinner="Memuat data...")
def load_all():
    d = {}
    with open(DATA_DIR+'config.json') as f: d['config'] = json.load(f)
    with open(DATA_DIR+'sys_config.json') as f: d['sys_config'] = json.load(f)
    df = pd.read_parquet(DATA_DIR+'all_scores.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    d['scores'] = df
    ev = pd.read_csv(DATA_DIR+'events.csv')
    ev['event_start'] = pd.to_datetime(ev['event_start'], utc=True)
    if 'event_end' in ev.columns: ev['event_end'] = pd.to_datetime(ev['event_end'], utc=True)
    d['events'] = ev
    d['thresholds'] = pd.read_csv(DATA_DIR+'thresholds.csv')
    theta = pd.read_csv(DATA_DIR+'theta_min_values.csv')
    d['theta_min'] = dict(zip(theta['score_col'], theta['theta_min']))
    for key, fname in {
        'sys_baseline':'system_eval_baseline.csv','sys_percoin':'system_eval_percoin.csv',
        'tradeoff':'tradeoff_curve_data.csv','comp_3way':'comparison_3way_percoin.csv',
        'summary_3way':'summary_3way.csv','tbl_48':'tabel_4_8_prauc_pooled.csv',
        'tbl_410':'tabel_4_10_classification_metrics.csv',
        'abl_thr_model':'ablation_threshold_model.csv','abl_thr_sys':'ablation_threshold_system.csv',
        'abl_bc':'ablation_budget_cooldown.csv','abl_ctx_model':'ablation_context_model.csv',
        'abl_ctx_sys':'ablation_context_system.csv','abl_ctx_percoin':'ablation_context_percoin.csv',
    }.items():
        p = DATA_DIR+fname
        d[key] = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    return d

try:
    D = load_all()
except Exception as e:
    st.error(f"**Data tidak ditemukan.** Pastikan folder `data/` berisi hasil export.\n\n`{e}`")
    st.stop()

CFG, SYSCFG, SCORES, EVENTS = D['config'], D['sys_config'], D['scores'], D['events']

# === HELPERS ===
C_XGB='#10b981';C_ZS='#f59e0b';C_FT='#6366f1';C_FT48='#3b82f6';C_RED='#ef4444';C_CYAN='#22d3ee';C_MUTE='#94a3b8'
LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(12,16,33,0.6)',
    font=dict(family='Outfit, sans-serif',color='#e2e8f0',size=12),
    xaxis=dict(gridcolor='#1e2748',zerolinecolor='#1e2748'),
    yaxis=dict(gridcolor='#1e2748',zerolinecolor='#1e2748'),
    margin=dict(l=50,r=20,t=55,b=45),
    hoverlabel=dict(bgcolor='#111631',font_size=12,font_family='JetBrains Mono'))

MODEL_DISPLAY = {
    'xgb_score': ('XGBoost', '#10b981'),
    'll_score': ('LL ZS (ctx=18)', '#fbbf24'),
    'll_ft_score': ('LL FT (ctx=18)', '#818cf8'),
    'll_score_ctx24': ('LL ZS (ctx=24)', '#d97706'),
    'll_ft_score_ctx24': ('LL FT (ctx=24)', '#6366f1'),
    'll_score_ctx48': ('LL ZS (ctx=48)', '#b45309'),
    'll_ft_score_ctx48': ('LL FT (ctx=48)', '#4f46e5'),
}

def mcard(label, value, sub="", delta=None, delta_label=""):
    dh = ""
    if delta is not None:
        cls = "m-delta-up" if delta >= 0 else "m-delta-down"
        sign = "+" if delta >= 0 else ""
        dh = f'<div class="{cls}">{sign}{delta:.4f} {delta_label}</div>'
    sh = f'<div class="m-sub">{sub}</div>' if sub else ""
    return f'<div class="m-card"><div class="m-label">{label}</div><div class="m-value">{value}</div>{sh}{dh}</div>'

def sec(t): st.markdown(f'<div class="sec-h">{t}</div>', unsafe_allow_html=True)
def sep(): st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
def insight(h): st.markdown(f'<div class="insight-box">{h}</div>', unsafe_allow_html=True)
def title(t, s=""):
    st.markdown(f'<div class="page-title">{t}</div>', unsafe_allow_html=True)
    if s: st.markdown(f'<div class="page-subtitle">{s}</div>', unsafe_allow_html=True)

# === SHARED SIMULATION ENGINE ===
def simulate_alerts(df_coin, score_col, theta, B, C_h):
    """Run alert simulation untuk satu koin."""
    df = df_coin.sort_values('timestamp').copy()
    df = df.dropna(subset=[score_col])
    last_alert_ts = None
    daily_count = defaultdict(int)
    results = []
    for _, row in df.iterrows():
        ts = row['timestamp']
        score = row[score_col]
        y = row['y']
        day_key = ts.strftime('%Y-%m-%d')
        is_alert = False
        suppression = None
        above = score >= theta
        if above:
            if daily_count[day_key] >= B:
                suppression = 'budget'
            elif last_alert_ts is not None:
                hours_since = (ts - last_alert_ts).total_seconds() / 3600
                if hours_since < C_h:
                    suppression = 'cooldown'
            if suppression is None:
                is_alert = True
                last_alert_ts = ts
                daily_count[day_key] += 1
        ratio = score / theta if theta > 0 else 0
        if is_alert: zone = 'ALERT'
        elif above and suppression: zone = f'SUPPRESSED ({suppression})'
        elif ratio >= 0.7: zone = 'ELEVATED'
        elif ratio >= 0.5: zone = 'WATCH'
        else: zone = 'NORMAL'
        results.append({
            'timestamp': ts, 'score': score, 'y': y,
            'above_theta': above, 'is_alert': is_alert,
            'suppression': suppression, 'zone': zone, 'score_ratio': ratio,
        })
    return pd.DataFrame(results)

def match_events_in_range(alerts_df, events_coin, lead_h=6):
    """Cek berapa event ter-hit oleh alert di range."""
    fired = alerts_df[alerts_df['is_alert']].copy()
    hits = []
    for _, ev in events_coin.iterrows():
        ev_ts = ev['event_start']
        window_start = ev_ts - pd.Timedelta(hours=lead_h)
        window_end = ev_ts - pd.Timedelta(hours=1)
        matching = fired[
            (fired['timestamp'] >= window_start) &
            (fired['timestamp'] <= window_end)
        ]
        is_hit = len(matching) > 0
        lead = (ev_ts - matching['timestamp'].min()).total_seconds() / 3600 if is_hit else np.nan
        hits.append({
            'event_start': ev_ts, 'is_hit': is_hit,
            'n_alerts_in_window': len(matching), 'lead_hours': lead,
        })
    hits_df = pd.DataFrame(hits)
    if 'lead_hours' in hits_df.columns:
        hits_df['lead_hours'] = pd.to_numeric(hits_df['lead_hours'], errors='coerce')
    return hits_df

# === SIDEBAR ===
with st.sidebar:
    st.markdown("""<div style="text-align:center;padding:20px 0 10px;">
        <div style="font-size:32px;">⚡</div>
        <div style="font-family:'Outfit';font-size:18px;font-weight:700;
                    background:linear-gradient(135deg,#6366f1,#22d3ee);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">Pump EWS</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:2px;">Evaluasi Komparatif Dashboard</div>
    </div>""", unsafe_allow_html=True)
    sep()
    page = st.radio("nav",["🏠 Overview","📈 Model-Level","🔔 System-Level","🔬 Ablation Study","🔍 Explorer","🎮 Simulasi"],label_visibility="collapsed")
    sep()
    st.markdown(f"""<div style="padding:0 8px;"><div style="font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;">Dataset Info</div>
    <div style="font-family:'JetBrains Mono';font-size:12px;color:#94a3b8;line-height:2;">
    Koin: <span style="color:#e2e8f0">{len(CFG['coins'])}</span><br>
    Periode: <span style="color:#e2e8f0">2023 – 2026</span><br>
    Baseline: <span style="color:#e2e8f0">P{int(CFG['baseline_quantile']*100)}</span><br>
    Lead: <span style="color:#e2e8f0">{CFG.get('lead_window_hours',6)}h</span><br>
    Budget: <span style="color:#e2e8f0">{SYSCFG['budget_B']}/day</span><br>
    Cooldown: <span style="color:#e2e8f0">{SYSCFG['cooldown_C_h']}h</span></div></div>""", unsafe_allow_html=True)
    sep()
    st.caption("Mohamad Galih Prasetyo Adi")
    st.caption("USB YPKP Bandung · 2026")

# ====================================================================
# PAGE: OVERVIEW
# ====================================================================
if page == "🏠 Overview":
    title("Pump Early Warning System","Evaluasi Komparatif XGBoost dan Lag-LLaMA pada Deteksi Pump Altcoin Mid-Cap")
    sep()
    test = SCORES[SCORES['split']=='test']
    prev = test['y'].mean()
    def safe_prauc(col):
        v = test.dropna(subset=[col])
        return average_precision_score(v['y'],v[col]) if (len(v)>0 and v['y'].sum()>0) else np.nan
    prauc = {k: safe_prauc(c) for k,c in [('xgb','xgb_score'),('ft18','ll_ft_score'),('ft48','ll_ft_score_ctx48'),('zs18','ll_score')] if c in test.columns}

    sec("Model Performance — PR-AUC (Test, P99)")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(mcard("XGBoost",f"{prauc.get('xgb',0):.4f}",f"22 fitur · {prauc.get('xgb',0)/prev:.1f}x lift"), unsafe_allow_html=True)
    with c2: st.markdown(mcard("LL Fine-Tuned 18h",f"{prauc.get('ft18',0):.4f}",f"univariate · {prauc.get('ft18',0)/prev:.1f}x lift",delta=prauc.get('ft18',0)-prauc.get('xgb',0),delta_label="vs XGB"), unsafe_allow_html=True)
    with c3: st.markdown(mcard("LL Fine-Tuned 48h",f"{prauc.get('ft48',0):.4f}",f"univariate · {prauc.get('ft48',0)/prev:.1f}x lift",delta=prauc.get('ft48',0)-prauc.get('xgb',0),delta_label="vs XGB"), unsafe_allow_html=True)
    with c4: st.markdown(mcard("Random Baseline",f"{prev:.4f}","prevalence rate"), unsafe_allow_html=True)

    sep()
    sec("Temuan Utama")
    ca,cb = st.columns(2)
    with ca: insight("<strong>🔑 Context Length = Game Changer</strong><br>LL FT ctx=48h mengungguli XGBoost di PR-AUC meski hanya 1 fitur vs 22. Gap baseline ctx=18h ternyata <strong>information asymmetry</strong>, bukan kelemahan arsitektur.")
    with cb: insight("<strong>📊 Budget Paling Sensitif</strong><br>Sensitivity ranking: Budget > Threshold > Cooldown > Context. Policy constraint mendominasi perilaku sistem lebih dari model choice.")

    sep()
    sec("Perbandingan Semua Konfigurasi")
    ctx_model = D.get('abl_ctx_model', pd.DataFrame())
    if len(ctx_model) > 0:
        fig = go.Figure()
        bar_data = []
        for _, row in ctx_model.iterrows():
            m,ctx,pr = row['Model'],row['Context_h'],row['PR-AUC']
            if m=='XGBoost': bar_data.append(('XGBoost<br>(22 fitur)',pr,C_XGB))
            elif m=='LL Zero-Shot': bar_data.append((f'LL ZS<br>ctx={ctx}h',pr,C_ZS))
            else: bar_data.append((f'LL FT<br>ctx={ctx}h',pr,'#818cf8' if ctx==18 else ('#6366f1' if ctx==24 else '#4f46e5')))
        names,vals,colors = zip(*bar_data)
        fig.add_trace(go.Bar(x=list(names),y=list(vals),marker_color=list(colors),
            text=[f"{v:.4f}" for v in vals],textposition='outside',
            textfont=dict(size=12,family='JetBrains Mono',color='#e2e8f0'),
            hovertemplate="<b>%{x}</b><br>PR-AUC: %{y:.4f}<extra></extra>"))
        fig.add_hline(y=prev,line_dash="dot",line_color=C_RED,line_width=1,
                      annotation_text=f"Random ({prev:.4f})",annotation_position="top left",
                      annotation_font=dict(size=10,color=C_MUTE))
        fig.update_layout(**LAYOUT,height=430,showlegend=False,title=dict(text="PR-AUC per Model & Context Length (Test, P99)",font=dict(size=15)))
        fig.update_yaxes(title_text="PR-AUC")
        st.plotly_chart(fig, use_container_width=True)

# ====================================================================
# PAGE: MODEL-LEVEL
# ====================================================================
elif page == "📈 Model-Level":
    title("Model-Level Evaluation","PR-AUC, Per-Koin, Context Ablation")
    sep()
    tab1,tab2,tab3 = st.tabs(["📊 PR-AUC Pooled","🪙 Per-Koin","📐 Context Ablation"])
    with tab1:
        sec("Tabel 4.8 — PR-AUC Pooled")
        t8 = D.get('tbl_48',pd.DataFrame())
        if len(t8)>0: st.dataframe(t8,use_container_width=True,hide_index=True)
        sec("Tabel 4.10 — Classification Metrics")
        t10 = D.get('tbl_410',pd.DataFrame())
        if len(t10)>0: st.dataframe(t10,use_container_width=True,hide_index=True)
    with tab2:
        sec("Per-Koin: 3-Model Comparison (ctx=18h)")
        comp = D.get('comp_3way',pd.DataFrame())
        if len(comp)>0:
            fig = go.Figure()
            for col,name,color in [('XGB','XGBoost',C_XGB),('LL_ZS','LL Zero-Shot',C_ZS),('LL_FT','LL Fine-Tuned',C_FT)]:
                if col in comp.columns:
                    fig.add_trace(go.Bar(x=comp['Symbol'],y=comp[col],name=name,marker_color=color,opacity=0.85))
            fig.update_layout(**LAYOUT,barmode='group',height=450,title="PR-AUC per Koin (Test, P99, ctx=18h)",yaxis_title="PR-AUC",
                              legend=dict(orientation='h',y=1.08,x=0.5,xanchor='center'))
            st.plotly_chart(fig,use_container_width=True)
            if 'XGB' in comp.columns and 'LL_FT' in comp.columns:
                sec("XGBoost vs Lag-LLaMA FT — Head-to-Head")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=comp['XGB'],y=comp['LL_FT'],mode='markers+text',
                    text=comp['Symbol'],textposition='top center',
                    textfont=dict(size=9,family='JetBrains Mono',color=C_MUTE),
                    marker=dict(size=10,color=C_FT,opacity=0.7,line=dict(width=1,color='#818cf8')),
                    hovertemplate="<b>%{text}</b><br>XGB: %{x:.4f}<br>LL_FT: %{y:.4f}<extra></extra>",showlegend=False))
                mx = max(comp['XGB'].max(),comp['LL_FT'].max())*1.1
                fig2.add_trace(go.Scatter(x=[0,mx],y=[0,mx],mode='lines',line=dict(dash='dot',color=C_MUTE,width=1),showlegend=False))
                fig2.update_layout(**LAYOUT,height=480,title="Di atas garis = LL FT unggul",xaxis_title="XGBoost PR-AUC",yaxis_title="LL FT PR-AUC")
                st.plotly_chart(fig2,use_container_width=True)
            st.dataframe(comp,use_container_width=True,hide_index=True)
    with tab3:
        sec("Context Length Ablation — Model Level")
        cm = D.get('abl_ctx_model',pd.DataFrame())
        if len(cm)>0:
            fig = go.Figure()
            for model,color in [('LL Zero-Shot',C_ZS),('LL Fine-Tuned',C_FT)]:
                md = cm[cm['Model']==model]; md = md[md['Context_h'].apply(lambda x:isinstance(x,(int,float)))].sort_values('Context_h')
                if len(md)>0:
                    fig.add_trace(go.Scatter(x=md['Context_h'],y=md['PR-AUC'],mode='lines+markers+text',name=model,
                        text=[f"{v:.4f}" for v in md['PR-AUC']],textposition='top center',
                        textfont=dict(size=11,family='JetBrains Mono'),line=dict(color=color,width=3),marker=dict(size=10)))
            xr = cm[cm['Model']=='XGBoost']
            if len(xr)>0:
                fig.add_hline(y=xr['PR-AUC'].iloc[0],line_dash="dash",line_color=C_XGB,line_width=2,
                             annotation_text=f"XGBoost ({xr['PR-AUC'].iloc[0]:.4f})",annotation_font=dict(size=11,color=C_XGB))
            fig.update_layout(**LAYOUT,height=460,title="PR-AUC vs Context Length",xaxis_title="Context Length (jam)",yaxis_title="PR-AUC")
            fig.update_xaxes(tickvals=[18,24,48])
            st.plotly_chart(fig,use_container_width=True)
            st.dataframe(cm,use_container_width=True,hide_index=True)
            insight("<strong>Interpretasi:</strong> Peningkatan context length secara konsisten meningkatkan PR-AUC Lag-LLaMA. Gap baseline bukan kelemahan arsitektur, melainkan <strong>information asymmetry</strong>.")

# ====================================================================
# PAGE: SYSTEM-LEVEL
# ====================================================================
elif page == "🔔 System-Level":
    title("System-Level Evaluation","Alert Simulation, Trade-off, Per-Koin Hit Rate")
    sep()
    tab1,tab2,tab3,tab4 = st.tabs(["📋 Baseline","⚖️ Trade-off","🪙 Per-Koin","📐 Context"])
    with tab1:
        sec(f"Tabel 4.11 — Baseline (B={SYSCFG['budget_B']}, C={SYSCFG['cooldown_C_h']}h, P99)")
        sb = D.get('sys_baseline',pd.DataFrame())
        if len(sb)>0:
            st.dataframe(sb,use_container_width=True,hide_index=True)
            cols = st.columns(len(sb))
            for i,(_,row) in enumerate(sb.iterrows()):
                model = row.get('Model',row.get('model',''))
                er,fa = row.get('EventRecall',0),row.get('FA_per_Day',0)
                color = C_XGB if 'XGB' in model or 'xgb' in model.lower() else (C_ZS if 'Zero' in model else C_FT)
                with cols[i]:
                    st.markdown(f'<div class="m-card" style="border-top:3px solid {color}"><div class="m-label">{model}</div><div class="m-value">{er:.4f}</div><div class="m-sub">EventRecall</div><div class="m-sub" style="margin-top:8px;">FA/Day: <span style="color:{C_RED}">{fa:.1f}</span></div></div>', unsafe_allow_html=True)
    with tab2:
        sec("Trade-off: EventRecall vs FA/Day")
        tdf = D.get('tradeoff',pd.DataFrame())
        if len(tdf)>0:
            mcol = 'model' if 'model' in tdf.columns else 'Model'
            fc = 'FA_per_Day' if 'FA_per_Day' in tdf.columns else 'fa_per_day'
            ec = 'EventRecall' if 'EventRecall' in tdf.columns else 'event_recall'
            fig = go.Figure()
            for m in tdf[mcol].unique():
                md = tdf[tdf[mcol]==m].sort_values(fc)
                color = C_XGB if 'xgb' in m.lower() else (C_ZS if 'zero' in m.lower() or 'zs' in m.lower() else C_FT)
                fig.add_trace(go.Scatter(x=md[fc],y=md[ec],mode='lines',name=m,line=dict(color=color,width=2.5)))
            fig.update_layout(**LAYOUT,height=500,title="Trade-off Curve",xaxis_title="FA/Day",yaxis_title="Event Recall",
                              legend=dict(orientation='h',y=1.08,x=0.5,xanchor='center'))
            st.plotly_chart(fig,use_container_width=True)
    with tab3:
        sec("Tabel 4.12 — Per-Koin Hit Rate")
        pc = D.get('sys_percoin',pd.DataFrame())
        if len(pc)>0: st.dataframe(pc,use_container_width=True,hide_index=True)
    with tab4:
        sec("Context Length Ablation — System Level")
        cs = D.get('abl_ctx_sys',pd.DataFrame())
        if len(cs)>0:
            st.dataframe(cs,use_container_width=True,hide_index=True)
            fig = make_subplots(rows=1,cols=2,subplot_titles=("EventRecall vs Context","FA/Day vs Context"))
            for model,color in [('LL Zero-Shot',C_ZS),('LL Fine-Tuned',C_FT)]:
                md = cs[(cs['Model']==model)&(cs['Context_h'].apply(lambda x:isinstance(x,(int,float))))].sort_values('Context_h')
                if len(md)>0:
                    fig.add_trace(go.Scatter(x=md['Context_h'],y=md['EventRecall'],mode='lines+markers',name=model,line=dict(color=color,width=2.5),marker=dict(size=9)),row=1,col=1)
                    fig.add_trace(go.Scatter(x=md['Context_h'],y=md['FA_per_Day'],mode='lines+markers',name=model,showlegend=False,line=dict(color=color,width=2.5,dash='dash'),marker=dict(size=9)),row=1,col=2)
            xr = cs[cs['Model']=='XGBoost']
            if len(xr)>0:
                fig.add_hline(y=xr['EventRecall'].iloc[0],line_dash="dash",line_color=C_XGB,row=1,col=1)
                fig.add_hline(y=xr['FA_per_Day'].iloc[0],line_dash="dash",line_color=C_XGB,row=1,col=2)
            fig.update_layout(**LAYOUT,height=430)
            fig.update_xaxes(title_text="Context (jam)",tickvals=[18,24,48])
            fig.update_yaxes(title_text="EventRecall",row=1,col=1)
            fig.update_yaxes(title_text="FA/Day",row=1,col=2)
            st.plotly_chart(fig,use_container_width=True)

# ====================================================================
# PAGE: ABLATION
# ====================================================================
elif page == "🔬 Ablation Study":
    title("Ablation Study","Sensitivitas Parameter: Threshold, Budget, Cooldown, Context")
    sep()
    tab1,tab2,tab3,tab4 = st.tabs(["📏 Threshold","💰 Budget & Cooldown","📐 Context","🏆 Ranking"])
    with tab1:
        sec("Ablation 1 — Variasi Threshold")
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Model-Level (Tabel 4.13)**")
            t13 = D.get('abl_thr_model',pd.DataFrame())
            if len(t13)>0: st.dataframe(t13,use_container_width=True,hide_index=True)
        with c2:
            st.markdown("**System-Level (Tabel 4.14)**")
            t14 = D.get('abl_thr_sys',pd.DataFrame())
            if len(t14)>0: st.dataframe(t14,use_container_width=True,hide_index=True)
    with tab2:
        sec("Ablation 2 — Budget & Cooldown")
        t15 = D.get('abl_bc',pd.DataFrame())
        if len(t15)>0:
            st.dataframe(t15,use_container_width=True,hide_index=True)
            if 'B' in t15.columns and 'C' in t15.columns:
                mcol = 'Model' if 'Model' in t15.columns else 'model'
                fig = make_subplots(rows=1,cols=2,subplot_titles=("Budget Sweep (C=6h)","Cooldown Sweep (B=3)"))
                for m in t15[mcol].unique():
                    color = C_XGB if 'xgb' in m.lower() else (C_ZS if 'Zero' in m else C_FT)
                    bd = t15[(t15[mcol]==m)&(t15['C']==6)].sort_values('B')
                    if len(bd)>0: fig.add_trace(go.Scatter(x=bd['B'],y=bd['EventRecall'],mode='lines+markers',name=m,line=dict(color=color,width=2.5),marker=dict(size=9)),row=1,col=1)
                    cd = t15[(t15[mcol]==m)&(t15['B']==3)].sort_values('C')
                    if len(cd)>0: fig.add_trace(go.Scatter(x=cd['C'],y=cd['EventRecall'],mode='lines+markers',name=m,showlegend=False,line=dict(color=color,width=2.5),marker=dict(size=9)),row=1,col=2)
                fig.update_layout(**LAYOUT,height=430)
                fig.update_xaxes(title_text="Budget (B)",row=1,col=1); fig.update_xaxes(title_text="Cooldown (C, jam)",row=1,col=2)
                fig.update_yaxes(title_text="EventRecall",row=1,col=1)
                st.plotly_chart(fig,use_container_width=True)
    with tab3:
        sec("Ablation 3 — Context Length")
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Model-Level (Tabel 4.16)**")
            acm = D.get('abl_ctx_model',pd.DataFrame())
            if len(acm)>0: st.dataframe(acm,use_container_width=True,hide_index=True)
        with c2:
            st.markdown("**System-Level (Tabel 4.17)**")
            acs = D.get('abl_ctx_sys',pd.DataFrame())
            if len(acs)>0: st.dataframe(acs,use_container_width=True,hide_index=True)
    with tab4:
        sec("Sensitivity Ranking — Mean ΔEventRecall")
        ranking = [("Budget (B=1→5)",0.3758,C_RED),("Threshold (P95→P99)",0.1897,C_ZS),("Cooldown (C=3→12h)",0.1757,C_FT),("Context (18→48h)",0.0423,C_CYAN)]
        fig = go.Figure()
        params = [r[0] for r in ranking][::-1]; deltas = [r[1] for r in ranking][::-1]; colors = [r[2] for r in ranking][::-1]
        fig.add_trace(go.Bar(y=params,x=deltas,orientation='h',marker=dict(color=colors),
            text=[f"  {d:.4f}" for d in deltas],textposition='outside',
            textfont=dict(family='JetBrains Mono',size=14,color='#e2e8f0')))
        fig.update_layout(**LAYOUT,height=320,showlegend=False,title="Sensitivity Ranking",xaxis_title="Mean ΔEventRecall")
        st.plotly_chart(fig,use_container_width=True)
        insight("<strong>Budget</strong> mendominasi karena langsung membatasi jumlah alert. <strong>Context Length</strong> ranking ke-4 di ΔER tapi dampak terbesar ada di <em>PR-AUC</em> — system-level ΔER teredam karena θ_min di-recalibrate per context.")

# ====================================================================
# PAGE: EXPLORER
# ====================================================================
elif page == "🔍 Explorer":
    title("Coin & Timeline Explorer","Visualisasi interaktif risk scores per koin")
    sep()
    c1,c2 = st.columns([1,3])
    with c1: coin = st.selectbox("Pilih Koin",CFG['coins'])
    test = SCORES[(SCORES['split']=='test')&(SCORES['symbol']==coin)].sort_values('timestamp')
    n_ev = int(test['y'].sum()) if len(test)>0 else 0
    with c2:
        st.markdown(f'<div style="padding:10px 0;"><span style="font-family:JetBrains Mono;font-size:24px;font-weight:700;color:#e2e8f0;">{coin}</span><span style="color:#94a3b8;margin-left:12px;">{len(test):,} timestamps · {n_ev} pump events (test)</span></div>', unsafe_allow_html=True)
    if len(test)==0: st.warning("Tidak ada data test."); st.stop()
    sep()
    score_opts = {k:v for k,v in MODEL_DISPLAY.items() if k in test.columns}
    defaults = [k for k in ['xgb_score','ll_ft_score','ll_ft_score_ctx48'] if k in score_opts]
    selected = st.multiselect("Model scores",list(score_opts.keys()),default=defaults,format_func=lambda x:score_opts[x][0])
    if not selected: st.info("Pilih minimal satu model."); st.stop()
    theta_dict = D.get('theta_min',{})
    show_theta = st.checkbox("Tampilkan θ_min threshold",value=True)
    sec(f"Timeline — {coin}")
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.25,0.75],subplot_titles=("Price","Risk Scores"),vertical_spacing=0.06)
    if 'close' in test.columns:
        fig.add_trace(go.Scatter(x=test['timestamp'],y=test['close'],mode='lines',line=dict(color='#475569',width=1),name='Close',showlegend=False),row=1,col=1)
    pumps = test[test['y']==1]
    for _,p in pumps.iterrows():
        fig.add_vrect(x0=p['timestamp']-pd.Timedelta(hours=1),x1=p['timestamp']+pd.Timedelta(hours=1),fillcolor=C_RED,opacity=0.08,line_width=0,row=2,col=1)
    if len(pumps)>0:
        fig.add_trace(go.Scatter(x=pumps['timestamp'],y=[1.02]*len(pumps),mode='markers',marker=dict(symbol='triangle-down',size=8,color=C_RED),name='Pump Event'),row=2,col=1)
    for col in selected:
        label,color = score_opts[col]
        valid = test.dropna(subset=[col])
        fig.add_trace(go.Scatter(x=valid['timestamp'],y=valid[col],mode='lines',line=dict(color=color,width=1.5),name=label),row=2,col=1)
        if show_theta and col in theta_dict:
            fig.add_hline(y=theta_dict[col],line_dash="dot",line_color=color,line_width=1,opacity=0.4,row=2,col=1)
    fig.update_layout(**LAYOUT,height=620,legend=dict(orientation='h',y=1.06,x=0.5,xanchor='center',font=dict(size=11)),title_text=f"{coin} — Test Period")
    fig.update_yaxes(title_text="Price ($)",row=1,col=1); fig.update_yaxes(title_text="Risk Score",row=2,col=1)
    st.plotly_chart(fig,use_container_width=True)
    sep()
    sec("🔎 Zoom ke Rentang Tanggal")
    c1,c2 = st.columns(2)
    min_d,max_d = test['timestamp'].min().date(),test['timestamp'].max().date()
    with c1: d1 = st.date_input("Dari",min_d,min_value=min_d,max_value=max_d,key="d1")
    with c2: d2 = st.date_input("Sampai",max_d,min_value=min_d,max_value=max_d,key="d2")
    if d1 < d2:
        zoom = test[(test['timestamp'].dt.date>=d1)&(test['timestamp'].dt.date<=d2)]
        if len(zoom)>0:
            fig2 = go.Figure()
            for col in selected:
                label,color = score_opts[col]; v = zoom.dropna(subset=[col])
                fig2.add_trace(go.Scatter(x=v['timestamp'],y=v[col],mode='lines',line=dict(color=color,width=2),name=label))
                if show_theta and col in theta_dict: fig2.add_hline(y=theta_dict[col],line_dash="dot",line_color=color,line_width=1,opacity=0.4)
            zp = zoom[zoom['y']==1]
            if len(zp)>0: fig2.add_trace(go.Scatter(x=zp['timestamp'],y=[1.02]*len(zp),mode='markers',marker=dict(symbol='triangle-down',size=10,color=C_RED),name='Pump Event'))
            fig2.update_layout(**LAYOUT,height=420,title_text=f"{coin}: {d1} → {d2}",yaxis_title="Risk Score",legend=dict(orientation='h',y=1.06,x=0.5,xanchor='center'))
            st.plotly_chart(fig2,use_container_width=True)
            st.caption(f"Zoom: {len(zoom):,} timestamps, {int(zoom['y'].sum())} pump events")

# ====================================================================
# PAGE: SIMULASI (UPDATED — 2 TABS)
# ====================================================================
elif page == "🎮 Simulasi":
    title("Simulation & Replay","Simulasi operasional dan multi-model head-to-head replay")
    sep()

    sim_tab1, sim_tab2 = st.tabs(["🎯 Single Model", "📊 Multi-Model Replay"])

    # ══════════════════════════════════════════════════════════════
    # TAB 1: SINGLE MODEL (existing)
    # ══════════════════════════════════════════════════════════════
    with sim_tab1:
        sec("Konfigurasi Simulasi")
        col_coin, col_model = st.columns(2)
        with col_coin:
            sim_coin = st.selectbox("Koin", CFG['coins'], key="sim_coin")
        with col_model:
            model_opts = {k: v[0] for k, v in MODEL_DISPLAY.items() if k in SCORES.columns}
            sim_model = st.selectbox("Model", list(model_opts.keys()),
                                     format_func=lambda x: model_opts[x], key="sim_model")
        sep()
        sec("Parameter Kebijakan Alert")
        col_t, col_b, col_c = st.columns(3)
        default_theta = D.get('theta_min', {}).get(sim_model, 0.5)
        with col_t:
            sim_theta = st.slider("θ (Threshold)", 0.01, 1.0, float(default_theta),
                                   step=0.01, format="%.4f",
                                   help="Skor minimum untuk trigger alert")
            st.caption(f"Default θ_min: {default_theta:.4f}")
        with col_b:
            sim_B = st.slider("B (Budget/hari)", 1, 10, SYSCFG['budget_B'],
                               help="Maks alert per koin per hari")
        with col_c:
            sim_C = st.slider("C (Cooldown, jam)", 1, 24, SYSCFG['cooldown_C_h'],
                               help="Jeda minimum antar alert")
        sep()
        sec("Rentang Waktu Simulasi")
        test_coin = SCORES[(SCORES['split'] == 'test') & (SCORES['symbol'] == sim_coin)].copy()
        test_coin = test_coin.sort_values('timestamp')
        if len(test_coin) == 0:
            st.warning(f"Tidak ada data test untuk {sim_coin}"); st.stop()
        min_dt = test_coin['timestamp'].min()
        max_dt = test_coin['timestamp'].max()
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            sim_start = st.date_input("Dari tanggal", min_dt.date(),
                                       min_value=min_dt.date(), max_value=max_dt.date(), key="sim_d1")
        with col_d2:
            sim_end = st.date_input("Sampai tanggal", min(min_dt.date() + pd.Timedelta(days=14), max_dt.date()),
                                     min_value=min_dt.date(), max_value=max_dt.date(), key="sim_d2")
        if sim_start >= sim_end:
            st.warning("Tanggal mulai harus sebelum tanggal akhir."); st.stop()
        range_data = test_coin[
            (test_coin['timestamp'].dt.date >= sim_start) &
            (test_coin['timestamp'].dt.date <= sim_end)
        ].copy()
        if len(range_data) == 0:
            st.warning("Tidak ada data di rentang ini."); st.stop()

        sim_result = simulate_alerts(range_data, sim_model, sim_theta, sim_B, sim_C)
        if len(sim_result) == 0:
            st.warning("Simulasi tidak menghasilkan data."); st.stop()
        sep()

        # Summary metrics
        sec("Hasil Simulasi")
        n_total = len(sim_result)
        n_alerts = int(sim_result['is_alert'].sum())
        n_above = int(sim_result['above_theta'].sum())
        n_suppressed = n_above - n_alerts
        n_days_sim = (pd.to_datetime(sim_end) - pd.to_datetime(sim_start)).days + 1
        events_coin = EVENTS[(EVENTS['symbol'] == sim_coin) & (EVENTS['quantile'] == CFG['baseline_quantile'])]
        events_in_range = events_coin[
            (events_coin['event_start'].dt.date >= sim_start) &
            (events_coin['event_start'].dt.date <= sim_end)
        ]
        if len(events_in_range) > 0:
            hits = match_events_in_range(sim_result, events_in_range, lead_h=CFG.get('lead_window_hours', 6))
            n_events_hit = int(hits['is_hit'].sum())
            n_events_total = len(hits)
            event_recall = n_events_hit / n_events_total if n_events_total > 0 else 0
        else:
            n_events_hit = n_events_total = 0; event_recall = 0; hits = pd.DataFrame()

        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: st.markdown(mcard("Alerts Fired",str(n_alerts),f"{n_alerts/n_days_sim:.1f}/hari" if n_days_sim>0 else ""),unsafe_allow_html=True)
        with c2: st.markdown(mcard("Suppressed",str(n_suppressed),"di atas θ tapi ditahan"),unsafe_allow_html=True)
        with c3: st.markdown(mcard("Near-Miss",str(int(sim_result['zone'].isin(['ELEVATED','WATCH']).sum())),"50-100% θ"),unsafe_allow_html=True)
        with c4: st.markdown(mcard("Events",f"{n_events_hit}/{n_events_total}",f"EventRecall: {event_recall:.2%}"),unsafe_allow_html=True)
        with c5: st.markdown(mcard("Timestamps",f"{n_total:,}",f"{n_days_sim} hari"),unsafe_allow_html=True)
        sep()

        # Timeline
        sec("Timeline Simulasi")
        fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.25,0.75],
                            subplot_titles=("Price","Risk Score & Zones"),vertical_spacing=0.06)
        if 'close' in range_data.columns:
            fig.add_trace(go.Scatter(x=range_data['timestamp'],y=range_data['close'],mode='lines',line=dict(color='#475569',width=1),name='Close',showlegend=False),row=1,col=1)
        model_color = C_FT if 'll_ft' in sim_model else (C_ZS if 'll_' in sim_model else C_XGB)
        fig.add_trace(go.Scatter(x=sim_result['timestamp'],y=sim_result['score'],mode='lines',name=model_opts.get(sim_model,sim_model),line=dict(color=model_color,width=2)),row=2,col=1)
        fig.add_hline(y=sim_theta,line_dash="dash",line_color=C_RED,line_width=1.5,annotation_text=f"θ = {sim_theta:.4f}",annotation_font=dict(size=10,color=C_RED),row=2,col=1)
        fig.add_hline(y=sim_theta*0.7,line_dash="dot",line_color=C_ZS,line_width=0.8,annotation_text="70% θ",annotation_font=dict(size=9,color=C_ZS),row=2,col=1)
        alerts_fired = sim_result[sim_result['is_alert']]
        if len(alerts_fired)>0:
            fig.add_trace(go.Scatter(x=alerts_fired['timestamp'],y=alerts_fired['score'],mode='markers',name='Alert Fired',
                marker=dict(symbol='star',size=14,color=C_RED,line=dict(width=1,color='white')),
                hovertemplate="<b>ALERT</b><br>%{x}<br>Score: %{y:.4f}<br>%{customdata}<extra></extra>",
                customdata=['✅ TP' if y==1 else '🔴 FP' for y in alerts_fired['y']]),row=2,col=1)
        suppressed = sim_result[sim_result['suppression'].notna()]
        if len(suppressed)>0:
            fig.add_trace(go.Scatter(x=suppressed['timestamp'],y=suppressed['score'],mode='markers',name='Suppressed',
                marker=dict(symbol='x',size=8,color='#f97316',opacity=0.6),
                hovertemplate="<b>SUPPRESSED (%{customdata})</b><br>%{x}<br>Score: %{y:.4f}<extra></extra>",
                customdata=suppressed['suppression']),row=2,col=1)
        for _,ev in events_in_range.iterrows():
            fig.add_vrect(x0=ev['event_start']-pd.Timedelta(hours=6),x1=ev['event_start'],fillcolor='rgba(239,68,68,0.1)',line_width=0,row=2,col=1)
            fig.add_vline(x=ev['event_start'],line_dash="solid",line_color=C_RED,line_width=1.5,opacity=0.7,row=2,col=1)
        fig.update_layout(**LAYOUT,height=600,legend=dict(orientation='h',y=1.06,x=0.5,xanchor='center',font=dict(size=11)),
                          title_text=f"{sim_coin} — {model_opts.get(sim_model,sim_model)} (θ={sim_theta:.4f}, B={sim_B}, C={sim_C}h)")
        fig.update_yaxes(title_text="Price",row=1,col=1); fig.update_yaxes(title_text="Risk Score",row=2,col=1)
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("""<div style="display:flex;gap:20px;flex-wrap:wrap;font-size:12px;color:#94a3b8;">
            <span>⭐ <span style="color:#ef4444">Alert Fired</span></span>
            <span>✕ <span style="color:#f97316">Suppressed</span></span>
            <span>| <span style="color:#ef4444">Pump event</span></span>
            <span>█ <span style="color:rgba(239,68,68,0.3)">Lead window</span></span>
        </div>""", unsafe_allow_html=True)
        sep()

        # Alert log
        sec("Log Alert")
        show_all = st.checkbox("Tampilkan semua timestamps", value=False)
        if show_all:
            display_df = sim_result[['timestamp','score','y','above_theta','is_alert','suppression','zone','score_ratio']].copy()
        else:
            display_df = sim_result[sim_result['zone'].isin(['ALERT','SUPPRESSED (budget)','SUPPRESSED (cooldown)','ELEVATED'])][['timestamp','score','y','above_theta','is_alert','suppression','zone','score_ratio']].copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['score'] = display_df['score'].round(6)
        display_df['score_ratio'] = (display_df['score_ratio']*100).round(1).astype(str)+'%'
        display_df = display_df.rename(columns={'timestamp':'Waktu','score':'Risk Score','y':'Pump?','above_theta':'Di atas θ','is_alert':'Alert?','suppression':'Suppression','zone':'Zone','score_ratio':'% θ'})
        st.dataframe(display_df,use_container_width=True,hide_index=True,height=400)
        st.caption(f"Menampilkan {len(display_df)} dari {len(sim_result)} timestamps")
        sep()

        # Event matching
        if len(events_in_range)>0 and len(hits)>0:
            sec("Event Matching — Hit / Miss")
            hd = hits.copy()
            hd['event_start'] = hd['event_start'].dt.strftime('%Y-%m-%d %H:%M')
            hd['lead_hours'] = pd.to_numeric(hd['lead_hours'], errors='coerce').round(1)
            hd['Status']=hd['is_hit'].map({True:'✅ HIT',False:'❌ MISS'})
            st.dataframe(hd[['event_start','Status','n_alerts_in_window','lead_hours']].rename(columns={'event_start':'Event Time','n_alerts_in_window':'Alerts in Window','lead_hours':'Lead (jam)'}),use_container_width=True,hide_index=True)
            insight(f"<strong>EventRecall:</strong> <em>{event_recall:.2%}</em> ({n_events_hit}/{n_events_total} events). Lead window: {CFG.get('lead_window_hours',6)}h.")

    # ══════════════════════════════════════════════════════════════
    # TAB 2: MULTI-MODEL REPLAY (NEW)
    # ══════════════════════════════════════════════════════════════
    with sim_tab2:
        insight("<strong>📊 Multi-Model Replay</strong><br>Bandingkan <em>semua model</em> secara simultan di setiap jam. Lihat mana yang fire alert, mana yang miss, dan mana yang false alarm — head-to-head.")
        sep()

        # ── Controls ──
        sec("Konfigurasi")
        mc1, mc2 = st.columns(2)
        with mc1:
            mm_coin = st.selectbox("Koin", CFG['coins'], key="mm_coin")
        with mc2:
            mm_models_available = {k: v[0] for k, v in MODEL_DISPLAY.items() if k in SCORES.columns}
            mm_defaults = [k for k in ['xgb_score', 'll_ft_score', 'll_ft_score_ctx48'] if k in mm_models_available]
            mm_selected = st.multiselect(
                "Model untuk dibandingkan", list(mm_models_available.keys()),
                default=mm_defaults, format_func=lambda x: mm_models_available[x], key="mm_models"
            )

        if not mm_selected:
            st.info("Pilih minimal satu model."); st.stop()

        # Policy
        sec("Parameter Kebijakan")
        pc1, pc2, pc3 = st.columns(3)
        theta_defaults = D.get('theta_min', {})
        with pc1:
            mm_B = st.slider("Budget (B/koin/hari)", 1, 10, SYSCFG['budget_B'], key="mm_B")
        with pc2:
            mm_C = st.slider("Cooldown (jam)", 1, 24, SYSCFG['cooldown_C_h'], key="mm_C")
        with pc3:
            mm_theta_mode = st.radio("θ_min mode", ["Per-model (default)", "Custom (sama semua)"], key="mm_theta_mode")

        if mm_theta_mode == "Custom (sama semua)":
            mm_theta_custom = st.slider("θ_min custom", 0.01, 1.0, 0.5, step=0.01, key="mm_theta_custom")
            mm_thetas = {m: mm_theta_custom for m in mm_selected}
        else:
            mm_thetas = {m: theta_defaults.get(m, 0.5) for m in mm_selected}
            theta_str = " · ".join([f"**{mm_models_available[m]}**: {mm_thetas[m]:.4f}" for m in mm_selected])
            st.caption(f"θ_min: {theta_str}")

        sep()

        # Date range
        sec("Rentang Waktu")
        mm_test = SCORES[(SCORES['split'] == 'test') & (SCORES['symbol'] == mm_coin)].sort_values('timestamp')
        if len(mm_test) == 0:
            st.warning(f"Tidak ada data test untuk {mm_coin}"); st.stop()
        mm_min = mm_test['timestamp'].min()
        mm_max = mm_test['timestamp'].max()
        dc1, dc2 = st.columns(2)
        with dc1:
            mm_d1 = st.date_input("Dari", mm_min.date(), min_value=mm_min.date(), max_value=mm_max.date(), key="mm_d1")
        with dc2:
            mm_d2 = st.date_input("Sampai", min(mm_min.date() + pd.Timedelta(days=7), mm_max.date()),
                                   min_value=mm_min.date(), max_value=mm_max.date(), key="mm_d2")
        if mm_d1 >= mm_d2:
            st.warning("Tanggal mulai harus sebelum tanggal akhir."); st.stop()

        mm_range = mm_test[
            (mm_test['timestamp'].dt.date >= mm_d1) & (mm_test['timestamp'].dt.date <= mm_d2)
        ].copy()
        if len(mm_range) == 0:
            st.warning("Tidak ada data di rentang ini."); st.stop()

        sep()

        # ── Run simulation for ALL models ──
        sec("Hasil Multi-Model")
        lead_h = CFG.get('lead_window_hours', 6)

        all_sim = {}
        for m_col in mm_selected:
            all_sim[m_col] = simulate_alerts(mm_range, m_col, mm_thetas[m_col], mm_B, mm_C)

        # Events in range
        mm_events = EVENTS[
            (EVENTS['symbol'] == mm_coin) & (EVENTS['quantile'] == CFG['baseline_quantile'])
        ]
        mm_events_range = mm_events[
            (mm_events['event_start'].dt.date >= mm_d1) & (mm_events['event_start'].dt.date <= mm_d2)
        ]

        # ── Build per-hour table ──
        timestamps = sorted(mm_range['timestamp'].unique())
        table_rows = []

        for ts in timestamps:
            row_data = {'Waktu': ts.strftime('%Y-%m-%d %H:%M')}
            ts_data = mm_range[mm_range['timestamp'] == ts]
            y_actual = int(ts_data['y'].iloc[0]) if len(ts_data) > 0 else 0
            row_data['🎯 Pump'] = '🎯 YA' if y_actual == 1 else ''

            for m_col in mm_selected:
                m_name = mm_models_available[m_col]
                sim = all_sim[m_col]
                m_row = sim[sim['timestamp'] == ts]

                if len(m_row) == 0:
                    row_data[f'{m_name} Score'] = '—'
                    row_data[f'{m_name} Status'] = '—'
                    row_data[f'{m_name} Verdict'] = '—'
                    continue

                m_row = m_row.iloc[0]
                score = m_row['score']
                is_alert = m_row['is_alert']
                suppression = m_row['suppression']
                above = m_row['above_theta']

                # Status
                if is_alert:
                    status = '✅ FIRED'
                elif above and suppression == 'budget':
                    status = '🚫 Budget'
                elif above and suppression == 'cooldown':
                    status = '⏸️ Cooldown'
                else:
                    ratio = score / mm_thetas[m_col] if mm_thetas[m_col] > 0 else 0
                    if ratio >= 0.7:
                        status = '🟡 Near'
                    else:
                        status = '❌ Low'

                # Verdict
                if is_alert and y_actual == 1:
                    verdict = '✅ TP'
                elif is_alert and y_actual == 0:
                    verdict = '🔴 FP'
                elif not is_alert and y_actual == 1:
                    verdict = '⚠️ MISS'
                else:
                    verdict = '—'

                row_data[f'{m_name} Score'] = f'{score:.4f}'
                row_data[f'{m_name} Status'] = status
                row_data[f'{m_name} Verdict'] = verdict

            table_rows.append(row_data)

        mm_table = pd.DataFrame(table_rows)

        # ── Filter ──
        filter_mode = st.radio(
            "Tampilkan",
            ["🔴 Hanya jam aktif (alert/event/near-miss)", "📋 Semua jam"],
            horizontal=True, key="mm_filter"
        )

        if filter_mode.startswith("🔴"):
            def has_activity(row):
                if row.get('🎯 Pump') == '🎯 YA':
                    return True
                for m_col in mm_selected:
                    m_name = mm_models_available[m_col]
                    status = row.get(f'{m_name} Status', '—')
                    if status not in ['❌ Low', '—']:
                        return True
                return False
            mm_filtered = mm_table[mm_table.apply(has_activity, axis=1)]
        else:
            mm_filtered = mm_table

        # ── Summary cards ──
        n_pumps_mm = (mm_table['🎯 Pump'] == '🎯 YA').sum()
        tp_counts = {}
        fp_counts = {}
        for m_col in mm_selected:
            m_name = mm_models_available[m_col]
            verdicts = mm_table[f'{m_name} Verdict']
            tp_counts[m_name] = (verdicts == '✅ TP').sum()
            fp_counts[m_name] = (verdicts == '🔴 FP').sum()

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(mcard("Jam Aktif", str(len(mm_filtered)), f"dari {len(mm_table)} total"), unsafe_allow_html=True)
        with mc2:
            st.markdown(mcard("Pump Events", str(n_pumps_mm), "di rentang ini"), unsafe_allow_html=True)
        with mc3:
            best = max(tp_counts, key=tp_counts.get) if tp_counts else '—'
            st.markdown(mcard("Best Detector", best, f"{max(tp_counts.values()) if tp_counts else 0} TP"), unsafe_allow_html=True)
        with mc4:
            cleanest = min(fp_counts, key=fp_counts.get) if fp_counts else '—'
            st.markdown(mcard("Fewest FP", cleanest, f"{min(fp_counts.values()) if fp_counts else 0} false alarms"), unsafe_allow_html=True)

        sep()

        # ── Per-hour table ──
        sec("Tabel Per-Jam — Head-to-Head")
        st.markdown("""<div style="font-size:12px;color:#94a3b8;margin-bottom:12px;line-height:1.8;">
            <strong>Status:</strong> ✅ FIRED · 🚫 Budget · ⏸️ Cooldown · 🟡 Near (70-100% θ) · ❌ Low<br>
            <strong>Verdict:</strong> ✅ TP = benar detect · 🔴 FP = false alarm · ⚠️ MISS = pump tak terdeteksi
        </div>""", unsafe_allow_html=True)

        st.dataframe(mm_filtered, use_container_width=True, hide_index=True, height=500)
        sep()

        # ── Multi-model score chart ──
        sec("Grafik Score — Semua Model")

        fig_mm = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.25, 0.75],
            subplot_titles=("Price", "Risk Scores — All Models"), vertical_spacing=0.06
        )

        if 'close' in mm_range.columns:
            fig_mm.add_trace(go.Scatter(x=mm_range['timestamp'], y=mm_range['close'],
                mode='lines', line=dict(color='#475569', width=1), name='Close', showlegend=False), row=1, col=1)

        for m_col in mm_selected:
            m_name = mm_models_available[m_col]
            color = MODEL_DISPLAY.get(m_col, ('', '#94a3b8'))[1]
            sim = all_sim[m_col]

            # Score line
            fig_mm.add_trace(go.Scatter(
                x=sim['timestamp'], y=sim['score'], mode='lines',
                name=m_name, line=dict(color=color, width=2), opacity=0.85
            ), row=2, col=1)

            # θ line
            fig_mm.add_hline(y=mm_thetas[m_col], line_dash="dot", line_color=color,
                             line_width=0.8, opacity=0.4, row=2, col=1)

            # TP markers
            alerts = sim[sim['is_alert']]
            tp = alerts[alerts['y'] == 1]
            fp = alerts[alerts['y'] == 0]
            if len(tp) > 0:
                fig_mm.add_trace(go.Scatter(
                    x=tp['timestamp'], y=tp['score'], mode='markers',
                    marker=dict(symbol='star', size=12, color=color, line=dict(width=2, color='#10b981')),
                    showlegend=False, hovertemplate=f"<b>{m_name} ✅ TP</b><br>%{{x}}<br>Score: %{{y:.4f}}<extra></extra>"
                ), row=2, col=1)
            if len(fp) > 0:
                fig_mm.add_trace(go.Scatter(
                    x=fp['timestamp'], y=fp['score'], mode='markers',
                    marker=dict(symbol='x', size=8, color=color, line=dict(width=2, color='#ef4444')),
                    showlegend=False, hovertemplate=f"<b>{m_name} 🔴 FP</b><br>%{{x}}<br>Score: %{{y:.4f}}<extra></extra>"
                ), row=2, col=1)

        # Event markers
        for _, ev in mm_events_range.iterrows():
            fig_mm.add_vrect(x0=ev['event_start'] - pd.Timedelta(hours=lead_h), x1=ev['event_start'],
                             fillcolor='rgba(239,68,68,0.08)', line_width=0, row=2, col=1)
            fig_mm.add_vline(x=ev['event_start'], line_dash="solid", line_color='#ef4444',
                             line_width=1.5, opacity=0.7, row=2, col=1)

        fig_mm.update_layout(**LAYOUT, height=600,
            legend=dict(orientation='h', y=1.06, x=0.5, xanchor='center', font=dict(size=10)),
            title_text=f"Multi-Model: {mm_coin} ({mm_d1} → {mm_d2})")
        fig_mm.update_yaxes(title_text="Price", row=1, col=1)
        fig_mm.update_yaxes(title_text="Risk Score", row=2, col=1)
        st.plotly_chart(fig_mm, use_container_width=True)

        st.markdown("""<div style="display:flex;gap:20px;flex-wrap:wrap;font-size:12px;color:#94a3b8;">
            <span>★ = Alert (border hijau = TP, merah = FP)</span>
            <span>| <span style="color:#ef4444">Garis merah</span> = Pump event</span>
            <span>█ <span style="color:rgba(239,68,68,0.3)">Area</span> = Lead window 6h</span>
            <span>┈ = θ_min per model</span>
        </div>""", unsafe_allow_html=True)
        sep()

        # ── Scorecard per model ──
        sec("Scorecard — Ringkasan Per Model")
        scorecard_cols = st.columns(len(mm_selected))

        for i, m_col in enumerate(mm_selected):
            m_name = mm_models_available[m_col]
            sim = all_sim[m_col]
            color = MODEL_DISPLAY.get(m_col, ('', '#94a3b8'))[1]

            n_fired = int(sim['is_alert'].sum())
            n_tp = int(((sim['is_alert']) & (sim['y'] == 1)).sum())
            n_fp = int(((sim['is_alert']) & (sim['y'] == 0)).sum())
            n_miss = int(((~sim['is_alert']) & (sim['y'] == 1)).sum())
            precision = n_tp / n_fired if n_fired > 0 else 0

            if len(mm_events_range) > 0:
                ev_hits = match_events_in_range(sim, mm_events_range, lead_h)
                er = ev_hits['is_hit'].mean() if len(ev_hits) > 0 else 0
                avg_lead = ev_hits[ev_hits['is_hit']]['lead_hours'].mean() if ev_hits['is_hit'].sum() > 0 else 0
            else:
                er = avg_lead = 0

            with scorecard_cols[i]:
                st.markdown(f"""
                <div class="m-card" style="border-top:3px solid {color};">
                    <div class="m-label">{m_name}</div>
                    <div style="font-family:'JetBrains Mono';font-size:12px;color:#e2e8f0;line-height:2.2;margin-top:8px;">
                        θ_min: <span style="color:{C_CYAN}">{mm_thetas[m_col]:.4f}</span><br>
                        Alerts: <span style="font-weight:700;">{n_fired}</span><br>
                        <span style="color:#10b981;">✅ TP: {n_tp}</span> ·
                        <span style="color:#ef4444;">🔴 FP: {n_fp}</span><br>
                        <span style="color:#f59e0b;">⚠️ Miss: {n_miss}</span><br>
                        Precision: <span style="color:{C_CYAN}">{precision:.1%}</span><br>
                        EventRecall: <span style="color:{C_CYAN}">{er:.1%}</span><br>
                        Avg Lead: <span style="color:{C_CYAN}">{avg_lead:.1f}h</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        sep()

        # ── Explainer ──
        with st.expander("ℹ️ Cara Membaca View Ini"):
            st.markdown("""
**Tabel Per-Jam** menampilkan setiap jam dalam rentang waktu yang dipilih.
Untuk setiap model, Anda bisa melihat:

- **Score**: Skor risiko (semakin tinggi = semakin yakin ada pump)
- **Status**: Apa yang terjadi pada alert
    - ✅ FIRED = alert terkirim
    - 🚫 Budget = skor cukup tinggi tapi kuota habis
    - ⏸️ Cooldown = skor cukup tinggi tapi masih jeda
    - 🟡 Near = skor 70-100% dari threshold
    - ❌ Low = skor di bawah threshold
- **Verdict**: Evaluasi kebenaran
    - ✅ TP = benar mendeteksi pump
    - 🔴 FP = false alarm
    - ⚠️ MISS = pump tidak terdeteksi

**Grafik** menampilkan skor semua model bersamaan.
★ = alert fired (border hijau = TP, merah = FP).
Area merah muda = lead window 6h sebelum pump.

**Scorecard** merangkum performa: TP, FP, miss, precision, dan EventRecall.

**Tips**: Gunakan filter "Hanya jam aktif" untuk fokus pada momen penting.
Geser θ_min ke mode Custom untuk lihat dampak threshold yang sama pada semua model.
            """)
