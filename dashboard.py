"""
dashboard.py — Tableau de bord interactif NLP × Finances Publiques Cameroun
Lancer : streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import ShuffleSplit, learning_curve
from sklearn.metrics import make_scorer, f1_score

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Audit Sémantique — Lois de Finances Cameroun",
    page_icon="🇨🇲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS PERSONNALISÉ
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F0F4F8; }
    .block-container { padding: 1.5rem 2rem; }

    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #1F3864;
        margin-bottom: 1rem;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1F3864; }
    .kpi-label { font-size: 0.85rem; color: #666; margin-top: 0.2rem; }
    .kpi-delta { font-size: 0.8rem; margin-top: 0.3rem; }

    .section-title {
        font-size: 1.25rem; font-weight: 700;
        color: #1F3864; border-bottom: 2px solid #1F3864;
        padding-bottom: 0.4rem; margin: 1.5rem 0 1rem 0;
    }

    /* ── Barre de filtres ── */
    .filter-bar {
        background: white;
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
        border-left: 5px solid #2E86C1;
    }
    .filter-label {
        font-size: 0.75rem; font-weight: 700; color: #2E86C1;
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.3rem;
    }
    .filter-summary {
        background: #EBF5FB; border-radius: 8px;
        padding: 0.45rem 1rem; font-size: 0.82rem;
        color: #2874A6; margin-bottom: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 94
    scores_sim = np.concatenate([
        np.random.normal(0.984, 0.008, 70),
        np.random.normal(0.955, 0.018, 18),
        np.random.uniform(0.862, 0.935, 6),
    ])
    scores_sim = np.clip(scores_sim, 0.862, 1.0)
    df_sim = pd.DataFrame({
        "Article":   [f"Article {i+1}" for i in range(n)],
        "Score_Max": np.sort(scores_sim)[::-1],
    })

    piliers = ["Transformation structurelle", "Gouvernance",
               "Développement régional", "Capital humain"]
    rows_2024, rows_2025 = [], []
    config = {
        "Transformation structurelle": (118, 157, 0.62, 0.08, 0.65, 0.07),
        "Gouvernance":                 ( 54,  13, 0.74, 0.10, 0.47, 0.06),
        "Développement régional":      (  2,   1, 0.52, 0.05, 0.52, 0.04),
        "Capital humain":              (  1,   8, 0.38, 0.03, 0.41, 0.04),
    }
    for pil, (n24, n25, mu24, s24, mu25, s25) in config.items():
        for _ in range(n24):
            rows_2024.append({"pilier": pil,
                              "score_pilier": np.clip(np.random.normal(mu24, s24), 0.1, 1.05),
                              "AE": np.random.lognormal(15.5, 1.5),
                              "CP": np.random.lognormal(15.2, 1.4)})
        for _ in range(n25):
            rows_2025.append({"pilier": pil,
                              "score_pilier": np.clip(np.random.normal(mu25, s25), 0.1, 1.05),
                              "AE": np.random.lognormal(15.5, 1.5),
                              "CP": np.random.lognormal(15.2, 1.4)})

    df_2024 = pd.DataFrame(rows_2024)
    df_2024["Année"] = 2024
    df_2024["Pilier_SND30"] = df_2024["pilier"]
    df_2025 = pd.DataFrame(rows_2025)
    df_2025["Année"] = 2025
    df_2025["Pilier_SND30"] = df_2025["pilier"]
    df_total = pd.concat([df_2024, df_2025], ignore_index=True)

    bilan = df_total.groupby("Pilier_SND30").agg(
        Total_AE=("AE", "sum"), Total_CP=("CP", "sum"),
        Frequence_Thematique=("AE", "count")
    ).reset_index()
    bilan["AE_Mrd"] = bilan["Total_AE"] / 1e9
    bilan["CP_Mrd"] = bilan["Total_CP"] / 1e9
    bilan["Pilier_Moteur"]       = bilan["Pilier_SND30"]
    bilan["Rang_AE"]             = bilan["AE_Mrd"].rank(ascending=False)
    bilan["Rang_Sémantique"]     = bilan["Frequence_Thematique"].rank(ascending=False)
    bilan["Poids_Frequence_%"]   = (bilan["Frequence_Thematique"] / bilan["Frequence_Thematique"].sum() * 100).round(1)
    bilan["Poids_Financier_%"]   = (bilan["AE_Mrd"] / bilan["AE_Mrd"].sum() * 100).round(1)

    df_js = pd.DataFrame({
        "Pilier":         ["Gouvernance", "Transformation structurelle", "Capital humain", "Développement régional"],
        "JS_Divergence":  [0.6558, 0.2362, None, None],
        "n_2024":         [54, 118, 1, 2],
        "n_2025":         [13, 157, 8, 1],
        "Intensite":      ["Forte", "Modérée", "N/A", "N/A"],
    })

    df_tests = pd.DataFrame({
        "Pilier":           piliers,
        "p_value_MW":       [0.0001, 0.0084, None, None],
        "taille_effet_r":   [0.486, 0.159, None, None],
        "Significatif":     ["✅ Oui", "✅ Oui", "N/A", "N/A"],
    })

    return df_sim, df_2024, df_2025, df_total, bilan, df_js, df_tests

df_sim, df_2024_cl, df_2025_cl, df_total, bilan_piliers, df_js, df_tests = load_data()

PILIERS    = ["Transformation structurelle", "Gouvernance", "Développement régional", "Capital humain"]
ALL_ANNEES = [2024, 2025]
COLORS_PIL = {
    "Transformation structurelle": "#1F77B4",
    "Gouvernance":                 "#E74C3C",
    "Développement régional":      "#2ECC71",
    "Capital humain":              "#9B59B6",
}
COLORS_AN = {2024: "#2E86C1", 2025: "#E74C3C"}

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE  — filtres persistants entre pages
# ─────────────────────────────────────────────────────────────────────────────
if "annees_sel"  not in st.session_state:
    st.session_state["annees_sel"]  = [2024, 2025]
if "piliers_sel" not in st.session_state:
    st.session_state["piliers_sel"] = PILIERS.copy()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/4a/Flag_of_Cameroon.svg", width=80)
    st.markdown("## 🇨🇲 Audit NLP — SND30")
    st.markdown("**Lois de Finances 2024 & 2025**")
    st.divider()

    page = st.radio("📌 Navigation", [
        "🏠 Vue d'ensemble",
        "📐 Glissement Sémantique",
        "🗂️ Classification Zero-Shot",
        "📊 Analyse Statistique",
        "💰 Corrélation Discours-Budget",
        "🤖 Courbe d'Apprentissage",
    ])

    st.divider()
    st.markdown("### 🎛️ Filtres globaux")

    # Multiselects sidebar synchronisés
    annees_sb = st.multiselect(
        "📅 Années", ALL_ANNEES,
        default=st.session_state["annees_sel"],
        key="sb_annees"
    )
    piliers_sb = st.multiselect(
        "🎯 Piliers SND30", PILIERS,
        default=st.session_state["piliers_sel"],
        key="sb_piliers"
    )
    if st.button("🔄 Réinitialiser", use_container_width=True):
        st.session_state["annees_sel"]  = [2024, 2025]
        st.session_state["piliers_sel"] = PILIERS.copy()
        st.rerun()

    # Synchroniser sidebar → session_state
    if set(annees_sb)  != set(st.session_state["annees_sel"]):
        st.session_state["annees_sel"]  = annees_sb
    if set(piliers_sb) != set(st.session_state["piliers_sel"]):
        st.session_state["piliers_sel"] = piliers_sb

    st.divider()
    st.caption("ISSEA · ISE3 Data Science · 2025-2026")

# ─────────────────────────────────────────────────────────────────────────────
# BARRE DE FILTRES RAPIDES (pills cliquables, affichée sur chaque page)
# ─────────────────────────────────────────────────────────────────────────────
PIL_SHORT = {
    "Transformation structurelle": "🏗️ Transform.",
    "Gouvernance":                 "⚖️ Gouvernance",
    "Développement régional":      "🌍 Rég. Dev.",
    "Capital humain":              "👩‍🎓 Cap. Humain",
}

def render_filter_bar():
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    st.markdown(
        '<div class="filter-label">⚡ Filtres rapides — cliquez pour activer / désactiver</div>',
        unsafe_allow_html=True
    )

    # Calcul des colonnes : label + 2 années + séparateur + 4 piliers + reset
    cols = st.columns([1.0, 0.85, 0.85, 0.15, 1.4, 1.4, 1.4, 1.4, 0.95])

    cols[0].markdown("**📅 Années**")

    # Pills Années
    for i, annee in enumerate(ALL_ANNEES):
        is_on = annee in st.session_state["annees_sel"]
        lbl   = f"{'✅' if is_on else '◻️'} {annee}"
        if cols[i + 1].button(lbl, key=f"pill_an_{annee}", use_container_width=True):
            if is_on and len(st.session_state["annees_sel"]) > 1:
                st.session_state["annees_sel"].remove(annee)
            elif not is_on:
                st.session_state["annees_sel"].append(annee)
            st.rerun()

    # Séparateur visuel
    cols[3].markdown("<div style='margin-top:8px;border-left:2px solid #ddd;height:28px'></div>",
                     unsafe_allow_html=True)

    # Pills Piliers
    for j, pil in enumerate(PILIERS):
        is_on = pil in st.session_state["piliers_sel"]
        lbl   = f"{'✅' if is_on else '◻️'} {PIL_SHORT[pil]}"
        if cols[j + 4].button(lbl, key=f"pill_pil_{pil}", use_container_width=True):
            if is_on and len(st.session_state["piliers_sel"]) > 1:
                st.session_state["piliers_sel"].remove(pil)
            elif not is_on:
                st.session_state["piliers_sel"].append(pil)
            st.rerun()

    # Bouton reset
    if cols[8].button("🔄 Reset", key="bar_reset", use_container_width=True):
        st.session_state["annees_sel"]  = [2024, 2025]
        st.session_state["piliers_sel"] = PILIERS.copy()
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Bandeau récapitulatif
    n_sel = len(df_total[
        df_total["Année"].isin(st.session_state["annees_sel"]) &
        df_total["Pilier_SND30"].isin(st.session_state["piliers_sel"])
    ])
    ann_str = " | ".join(f"**{a}**" for a in sorted(st.session_state["annees_sel"]))
    pil_str = " · ".join(st.session_state["piliers_sel"])
    st.markdown(
        f'<div class="filter-summary">🔎 Actif — Années : {ann_str} &nbsp;│&nbsp; '
        f'Piliers : {pil_str} &nbsp;│&nbsp; <b>{n_sel} lignes budgétaires</b></div>',
        unsafe_allow_html=True
    )

# Raccourcis
def A(): return st.session_state["annees_sel"]
def P(): return st.session_state["piliers_sel"]
def DF():
    return df_total[
        df_total["Année"].isin(A()) &
        df_total["Pilier_SND30"].isin(P())
    ]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER KPI
# ─────────────────────────────────────────────────────────────────────────────
def kpi(label, value, delta=None, delta_color="normal"):
    arrow = ""
    if delta:
        color = "#C0392B" if delta_color == "inverse" else "#1E8449"
        arrow = f'<div class="kpi-delta" style="color:{color}">{delta}</div>'
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {arrow}
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — VUE D'ENSEMBLE
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Vue d'ensemble":
    st.title("🇨🇲 Audit Sémantique des Lois de Finances")
    st.markdown("**Intelligence Artificielle et Finances Publiques — ISSEA 2025-2026**")
    st.divider()
    render_filter_bar()

    df_filt = DF()
    if df_filt.empty:
        st.warning("⚠️ Aucune donnée pour les filtres actifs.")
        st.stop()

    # KPIs dynamiques
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Lignes sélectionnées", f"{len(df_filt)}",
                 f"{len(A())} an · {df_filt['Pilier_SND30'].nunique()} pilier(s)")
    with c2: kpi("Score alignement moyen", f"{df_filt['score_pilier'].mean():.3f}")
    with c3: kpi("Total AE (Mrd FCFA)", f"{df_filt['AE'].sum()/1e9:.1f}")
    with c4: kpi("Total CP (Mrd FCFA)", f"{df_filt['CP'].sum()/1e9:.1f}")
    with c5: kpi("Div. JS — Gouvernance",
                 "0.656" if "Gouvernance" in P() else "N/A",
                 "⚠️ Rupture forte" if "Gouvernance" in P() else "Pilier non sélectionné",
                 "inverse")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">📊 Distribution lignes budgétaires par pilier</div>',
                    unsafe_allow_html=True)
        counts = df_filt.groupby(["Pilier_SND30", "Année"]).size().reset_index(name="count")
        fig = px.bar(counts, x="Pilier_SND30", y="count", color="Année",
                     barmode="group", color_discrete_map=COLORS_AN,
                     text="count",
                     labels={"Pilier_SND30": "Pilier", "count": "Lignes"})
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, plot_bgcolor="white", legend_title="Année")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">🎯 Score moyen d\'alignement par pilier</div>',
                    unsafe_allow_html=True)
        sc = df_filt.groupby(["Pilier_SND30", "Année"])["score_pilier"].mean().reset_index()
        fig2 = px.bar(sc, x="Pilier_SND30", y="score_pilier", color="Année",
                      barmode="group", color_discrete_map=COLORS_AN,
                      range_y=[0, 1], text=sc["score_pilier"].round(3),
                      labels={"score_pilier": "Score moyen", "Pilier_SND30": "Pilier"})
        fig2.update_traces(textposition="outside", texttemplate="%{text:.3f}")
        fig2.update_layout(height=380, plot_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    # AE vs CP par pilier
    st.markdown('<div class="section-title">💰 AE vs CP par pilier (Mrd FCFA)</div>',
                unsafe_allow_html=True)
    agg = df_filt.groupby("Pilier_SND30").agg(AE=("AE","sum"), CP=("CP","sum")).reset_index()
    agg["AE_Mrd"] = agg["AE"]/1e9
    agg["CP_Mrd"] = agg["CP"]/1e9
    fig_ae = go.Figure()
    fig_ae.add_trace(go.Bar(x=agg["Pilier_SND30"], y=agg["AE_Mrd"],
                            name="AE", marker_color="#2E86C1",
                            text=agg["AE_Mrd"].round(1), textposition="outside"))
    fig_ae.add_trace(go.Bar(x=agg["Pilier_SND30"], y=agg["CP_Mrd"],
                            name="CP", marker_color="#E74C3C",
                            text=agg["CP_Mrd"].round(1), textposition="outside"))
    fig_ae.update_layout(barmode="group", height=340, plot_bgcolor="white",
                         yaxis_title="Milliards FCFA")
    st.plotly_chart(fig_ae, use_container_width=True)

    # Tableau JS filtré
    st.markdown('<div class="section-title">📋 Divergence Jensen-Shannon</div>',
                unsafe_allow_html=True)
    def badge(row):
        return {"Forte": "🔴 Forte", "Modérée": "🟡 Modérée"}.get(row["Intensite"], "⚪ N/A")
    dj = df_js[df_js["Pilier"].isin(P())].copy()
    dj["Intensité"]     = dj.apply(badge, axis=1)
    dj["JS_Divergence"] = dj["JS_Divergence"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    st.dataframe(dj[["Pilier","JS_Divergence","n_2024","n_2025","Intensité"]],
                 use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — GLISSEMENT SÉMANTIQUE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📐 Glissement Sémantique":
    st.title("📐 Indice de Volatilité Législative — CamemBERT")
    st.caption("Similarité cosinus article par article entre LF 2024 et LF 2025")
    st.divider()
    render_filter_bar()

    seuil = st.slider("Seuil de rupture sémantique", 0.80, 0.99, 0.95, 0.01)
    df_sorted = df_sim.sort_values("Score_Max", ascending=False).reset_index(drop=True)
    df_sorted["Couleur"] = df_sorted["Score_Max"].apply(
        lambda x: "Rupture" if x < seuil else ("Stable" if x >= 0.98 else "Modéré"))

    c1, c2, c3 = st.columns(3)
    with c1: kpi("Score moyen", f"{df_sorted['Score_Max'].mean():.4f}")
    with c2: kpi("Médiane",     f"{df_sorted['Score_Max'].median():.4f}")
    with c3: kpi("Ruptures détectées",
                 str((df_sorted["Score_Max"] < seuil).sum()), f"< {seuil}")

    st.divider()
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-title">📊 Scores par article (triés)</div>',
                    unsafe_allow_html=True)
        cmap = {"Rupture": "#E74C3C", "Modéré": "#F39C12", "Stable": "#1E8449"}
        fig = px.bar(df_sorted, x=df_sorted.index, y="Score_Max", color="Couleur",
                     color_discrete_map=cmap, hover_data=["Article","Score_Max"],
                     labels={"x": "Articles", "Score_Max": "Similarité cosinus"})
        fig.add_hline(y=df_sorted["Score_Max"].mean(), line_dash="dash",
                      line_color="#2E86C1",
                      annotation_text=f"Moy. {df_sorted['Score_Max'].mean():.3f}")
        fig.add_hline(y=seuil, line_dash="dot", line_color="#E74C3C",
                      annotation_text=f"Seuil {seuil}")
        fig.update_layout(height=400, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">📈 Distribution</div>',
                    unsafe_allow_html=True)
        fig2 = px.histogram(df_sorted, x="Score_Max", nbins=20,
                            color_discrete_sequence=["#2E86C1"])
        fig2.add_vline(x=seuil, line_color="red", line_dash="dash",
                       annotation_text=f"Seuil={seuil}")
        fig2.update_layout(height=400, plot_bgcolor="white", bargap=0.05)
        st.plotly_chart(fig2, use_container_width=True)

    # JS filtré par piliers actifs
    st.markdown('<div class="section-title">📉 Divergence Jensen-Shannon par pilier</div>',
                unsafe_allow_html=True)
    dj = df_js[df_js["Pilier"].isin(P())].dropna(subset=["JS_Divergence"])
    if dj.empty:
        st.info("💡 Sélectionnez Gouvernance ou Transformation structurelle pour voir la divergence JS.")
    else:
        fig3 = px.bar(dj, x="JS_Divergence", y="Pilier", orientation="h",
                      color="Intensite",
                      color_discrete_map={"Forte": "#E74C3C", "Modérée": "#F39C12"},
                      text="JS_Divergence")
        fig3.add_vline(x=0.15, line_dash="dash", line_color="orange",
                       annotation_text="Seuil modéré 0.15")
        fig3.add_vline(x=0.30, line_dash="dash", line_color="red",
                       annotation_text="Seuil fort 0.30")
        fig3.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig3.update_layout(height=max(180, len(dj)*90), plot_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CLASSIFICATION ZERO-SHOT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🗂️ Classification Zero-Shot":
    st.title("🗂️ Classification Zero-Shot — Piliers SND30")
    st.caption("Alignement des lignes budgétaires via BART-large-MNLI")
    st.divider()
    render_filter_bar()

    df_filt = DF()
    if df_filt.empty:
        st.warning("⚠️ Aucune donnée pour les filtres actifs."); st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">📊 Répartition des piliers</div>',
                    unsafe_allow_html=True)
        counts = df_filt.groupby(["Pilier_SND30","Année"]).size().reset_index(name="count")
        fig = px.bar(counts, x="Pilier_SND30", y="count", color="Année",
                     barmode="group", text="count",
                     color_discrete_map=COLORS_AN)
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, plot_bgcolor="white", xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">🎻 Violin — Scores de confiance</div>',
                    unsafe_allow_html=True)
        if df_filt["score_pilier"].notna().sum() >= 4:
            fig2 = px.violin(df_filt, x="Pilier_SND30", y="score_pilier", color="Année",
                             box=True, points=False, color_discrete_map=COLORS_AN,
                             labels={"score_pilier": "Score", "Pilier_SND30": "Pilier"})
            fig2.update_layout(height=380, plot_bgcolor="white", xaxis_tickangle=-30)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Pas assez de données.")

    st.markdown('<div class="section-title">📦 Boxplot scores × pilier × année</div>',
                unsafe_allow_html=True)
    fig_box = px.box(df_filt, x="Pilier_SND30", y="score_pilier", color="Année",
                     color_discrete_map=COLORS_AN, notched=True, points="outliers",
                     labels={"score_pilier": "Score", "Pilier_SND30": "Pilier"})
    fig_box.update_layout(height=360, plot_bgcolor="white", xaxis_tickangle=-20)
    st.plotly_chart(fig_box, use_container_width=True)

    # Densité par année sélectionnée
    st.markdown('<div class="section-title">📈 Densité des scores</div>',
                unsafe_allow_html=True)
    if A():
        kde_cols = st.columns(len(A()))
        for i, annee in enumerate(sorted(A())):
            df_a = df_filt[df_filt["Année"] == annee]
            fig3 = go.Figure()
            for pil in P():
                vals = df_a.loc[df_a["Pilier_SND30"] == pil, "score_pilier"].dropna()
                if len(vals) >= 3:
                    from scipy.stats import gaussian_kde
                    xs  = np.linspace(0, 1.2, 200)
                    kde = gaussian_kde(vals, bw_method=0.3)
                    fig3.add_trace(go.Scatter(
                        x=xs, y=kde(xs), mode="lines", name=pil,
                        line=dict(color=COLORS_PIL.get(pil,"#999"), width=2.5),
                        fill="tozeroy", fillcolor="rgba(0,0,0,0.04)"))
            fig3.update_layout(title=f"Densité — {annee}", height=300,
                               plot_bgcolor="white", xaxis_title="Score",
                               legend=dict(orientation="h", y=-0.3))
            kde_cols[i].plotly_chart(fig3, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ANALYSE STATISTIQUE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analyse Statistique":
    st.title("📊 Tests de Significativité — Évolution Thématique SND30")
    st.divider()
    render_filter_bar()

    df_filt = DF()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Mann-Whitney U")
        st.dataframe(
            df_tests[df_tests["Pilier"].isin(P())][
                ["Pilier","p_value_MW","taille_effet_r","Significatif"]
            ].fillna("N/A"),
            use_container_width=True, hide_index=True
        )
        st.caption("H = 21.677, p = 0.0001 (Kruskal-Wallis global)")

    with col2:
        st.markdown("#### Chi² (Année × Pilier)")
        st.metric("χ²", "36.358")
        st.metric("p-value", "0.0000")
        st.metric("ddl", "3")
        st.success("✅ Significatif — Rejet H₀")

    with col3:
        st.markdown("#### Kruskal-Wallis")
        st.metric("H", "21.677")
        st.metric("p-value", "0.0001")
        st.metric("ddl", "3")
        st.success("✅ Significatif")

    st.divider()
    col4, col5 = st.columns(2)

    with col4:
        st.markdown('<div class="section-title">📉 P-values Mann-Whitney</div>',
                    unsafe_allow_html=True)
        dp = df_tests[df_tests["Pilier"].isin(P())].dropna(subset=["p_value_MW"])
        if dp.empty:
            st.info("Données disponibles pour Gouvernance & Transformation.")
        else:
            fig = px.bar(dp, x="p_value_MW", y="Pilier", orientation="h",
                         color="Pilier", color_discrete_map=COLORS_PIL,
                         text="p_value_MW")
            fig.add_vline(x=0.05, line_dash="dash", line_color="black",
                          annotation_text="α = 0.05")
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(height=300, plot_bgcolor="white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col5:
        st.markdown('<div class="section-title">🔥 Heatmap Chi² — Fréquences observées</div>',
                    unsafe_allow_html=True)
        raw = {
            "Capital humain":              [1, 8],
            "Développement régional":      [2, 1],
            "Gouvernance":                 [54, 13],
            "Transformation structurelle": [118, 157],
        }
        obs_full = pd.DataFrame(raw, index=[2024, 2025])
        obs = obs_full.loc[
            [a for a in [2024, 2025] if a in A()],
            [p for p in PILIERS if p in P()]
        ]
        if obs.empty:
            st.info("Sélectionnez au moins une année et un pilier.")
        else:
            fig2 = px.imshow(obs, text_auto=True, color_continuous_scale="Reds",
                             labels={"x":"Pilier","y":"Année","color":"Fréquence"},
                             aspect="auto")
            fig2.update_layout(height=max(200, len(obs)*100))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">📦 Distribution AE par pilier × année (log)</div>',
                unsafe_allow_html=True)
    if df_filt.empty:
        st.warning("Aucune donnée.")
    else:
        fig3 = px.box(df_filt, x="Pilier_SND30", y="AE", color="Année",
                      color_discrete_map=COLORS_AN, log_y=True,
                      labels={"AE":"Montant AE (log FCFA)", "Pilier_SND30":"Pilier"})
        fig3.update_layout(height=380, plot_bgcolor="white", xaxis_tickangle=-20)
        st.plotly_chart(fig3, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CORRÉLATION DISCOURS-BUDGET
# ═════════════════════════════════════════════════════════════════════════════
elif page == "💰 Corrélation Discours-Budget":
    st.title("💰 Corrélation Visibilité Sémantique vs Effort Financier")
    st.divider()
    render_filter_bar()

    df_filt = DF()
    if df_filt.empty:
        st.warning("⚠️ Aucune donnée."); st.stop()

    # Bilan recalculé sur la sélection courante (années + piliers)
    bilan_dyn = df_filt.groupby("Pilier_SND30").agg(
        Total_AE=("AE","sum"), Total_CP=("CP","sum"),
        Frequence_Thematique=("AE","count")
    ).reset_index()
    bilan_dyn["AE_Mrd"] = bilan_dyn["Total_AE"] / 1e9
    bilan_dyn["CP_Mrd"] = bilan_dyn["Total_CP"] / 1e9
    bilan_dyn["Pilier_Moteur"]     = bilan_dyn["Pilier_SND30"]
    bilan_dyn["Rang_AE"]           = bilan_dyn["AE_Mrd"].rank(ascending=False)
    bilan_dyn["Rang_Sémantique"]   = bilan_dyn["Frequence_Thematique"].rank(ascending=False)
    tf = bilan_dyn["Frequence_Thematique"].sum()
    ta = bilan_dyn["AE_Mrd"].sum()
    bilan_dyn["Poids_Frequence_%"] = (bilan_dyn["Frequence_Thematique"]/tf*100).round(1) if tf else 0
    bilan_dyn["Poids_Financier_%"] = (bilan_dyn["AE_Mrd"]/ta*100).round(1) if ta else 0

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">📈 Régression Fréquence vs CP</div>',
                    unsafe_allow_html=True)
        fig = px.scatter(bilan_dyn, x="Frequence_Thematique", y="CP_Mrd",
                         text="Pilier_SND30", color="Pilier_SND30",
                         color_discrete_map=COLORS_PIL,
                         trendline="ols" if len(bilan_dyn) >= 3 else None,
                         labels={"Frequence_Thematique":"Fréquence", "CP_Mrd":"CP (Mrd)"})
        fig.update_traces(textposition="top center")
        fig.update_layout(height=400, plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">🎯 Arbitrage Fréquence vs AE</div>',
                    unsafe_allow_html=True)
        fig2 = px.scatter(bilan_dyn, x="Frequence_Thematique", y="AE_Mrd",
                          text="Pilier_SND30", color="Pilier_SND30",
                          color_discrete_map=COLORS_PIL,
                          trendline="ols" if len(bilan_dyn) >= 3 else None,
                          labels={"Frequence_Thematique":"Fréquence", "AE_Mrd":"AE (Mrd)"})
        fig2.update_traces(textposition="top center")
        fig2.update_layout(height=400, plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-title">📊 Volume Projets vs CP</div>',
                    unsafe_allow_html=True)
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Bar(x=bilan_dyn["Pilier_Moteur"], y=bilan_dyn["Frequence_Thematique"],
                              name="Nb projets", marker_color="lightblue"), secondary_y=False)
        fig3.add_trace(go.Scatter(x=bilan_dyn["Pilier_Moteur"], y=bilan_dyn["CP_Mrd"],
                                  name="CP (Mrd)", mode="lines+markers",
                                  marker=dict(color="#1F3864", size=12),
                                  line=dict(width=2.5)), secondary_y=True)
        fig3.update_layout(height=380, plot_bgcolor="white", xaxis_tickangle=-20)
        fig3.update_yaxes(title_text="Nb projets", secondary_y=False)
        fig3.update_yaxes(title_text="Mrd FCFA", secondary_y=True)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-title">🏹 Poids Fréquence vs Financier (%)</div>',
                    unsafe_allow_html=True)
        fig4 = go.Figure()
        for _, row in bilan_dyn.iterrows():
            fig4.add_shape(type="line",
                x0=row["Poids_Frequence_%"], y0=row["Pilier_Moteur"],
                x1=row["Poids_Financier_%"],  y1=row["Pilier_Moteur"],
                line=dict(color="lightgrey", dash="dot", width=1.5))
        fig4.add_trace(go.Scatter(
            x=bilan_dyn["Poids_Frequence_%"], y=bilan_dyn["Pilier_Moteur"],
            mode="markers", name="Fréquence (%)",
            marker=dict(color="#2E86C1", size=16)))
        fig4.add_trace(go.Scatter(
            x=bilan_dyn["Poids_Financier_%"], y=bilan_dyn["Pilier_Moteur"],
            mode="markers", name="Financier (%)",
            marker=dict(color="#E74C3C", size=16)))
        fig4.update_layout(height=380, plot_bgcolor="white", xaxis_title="Part (%)")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-title">🎯 Décalage des Rangs (Discours vs Budget)</div>',
                unsafe_allow_html=True)
    fig5 = go.Figure()
    for _, row in bilan_dyn.iterrows():
        fig5.add_shape(type="line",
            x0=row["Rang_Sémantique"], y0=row["Pilier_Moteur"],
            x1=row["Rang_AE"],         y1=row["Pilier_Moteur"],
            line=dict(color="lightgrey", width=2, dash="dot"))
    fig5.add_trace(go.Scatter(
        x=bilan_dyn["Rang_Sémantique"], y=bilan_dyn["Pilier_Moteur"],
        mode="markers", name="Rang Discours",
        marker=dict(color="#1F3864", size=18, opacity=0.7)))
    fig5.add_trace(go.Scatter(
        x=bilan_dyn["Rang_AE"], y=bilan_dyn["Pilier_Moteur"],
        mode="markers+text", name="Rang Budget",
        marker=dict(color="#E74C3C", size=14),
        text=bilan_dyn["Pilier_Moteur"], textposition="middle right"))
    fig5.update_xaxes(autorange="reversed", title="Rang (1 = priorité max)")
    fig5.update_layout(height=340, plot_bgcolor="white")
    st.plotly_chart(fig5, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — COURBE D'APPRENTISSAGE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Courbe d'Apprentissage":
    st.title("🤖 Courbe d'Apprentissage du Classificateur")
    st.caption("F1-Score (macro) en fonction de la taille du corpus d'entraînement")
    st.divider()
    render_filter_bar()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    df_filt = DF()
    if df_filt.empty:
        st.warning("⚠️ Aucune donnée."); st.stop()

    df_ml = df_filt.dropna(subset=["score_pilier","AE"])
    le    = LabelEncoder()
    y     = le.fit_transform(df_ml["Pilier_SND30"])
    X     = df_ml[["score_pilier","AE"]].values

    col_params, col_chart = st.columns([1, 3])

    with col_params:
        st.markdown("#### ⚙️ Paramètres")
        n_splits  = st.slider("n_splits (CV)", 5, 20, 10)
        test_size = st.slider("test_size", 0.1, 0.4, 0.2, 0.05)
        n_points  = st.slider("Points sur la courbe", 3, 8, 5)
        run_btn   = st.button("▶️ Lancer l'analyse", type="primary", use_container_width=True)
        st.divider()
        st.info(
            f"📦 Observations : **{len(X)}**\n\n"
            f"🏷️ Classes : **{len(le.classes_)}**\n\n"
            f"📅 Années : **{', '.join(map(str, sorted(A())))}**\n\n"
            f"🎯 Piliers : **{len(P())}**"
        )

    with col_chart:
        if run_btn or True:
            with st.spinner("Calcul…"):
                f1_sc = make_scorer(f1_score, average="macro", zero_division=0)
                cv    = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
                clf   = LogisticRegression(max_iter=1000, random_state=42)

                min_tr = max(len(le.classes_)*2, 10)
                max_tr = int(len(X)*(1-test_size))

                if min_tr >= max_tr:
                    st.warning("⚠️ Données insuffisantes. Sélectionnez plus de piliers ou les 2 années.")
                    st.stop()

                sizes = np.linspace(min_tr, max_tr, n_points).astype(int)

                try:
                    ts, tr_sc, te_sc = learning_curve(clf, X, y, train_sizes=sizes,
                                                      cv=cv, scoring=f1_sc, n_jobs=-1)
                    if np.isnan(tr_sc).all():
                        ts, tr_sc, te_sc = learning_curve(clf, X, y, train_sizes=sizes,
                                                          cv=cv, scoring="accuracy", n_jobs=-1)
                        metric_label = "Accuracy"
                        st.warning("⚠️ F1 vide — fallback Accuracy")
                    else:
                        metric_label = "F1-Score (macro)"

                    trm, trs = tr_sc.mean(1), tr_sc.std(1)
                    tem, tes = te_sc.mean(1), te_sc.std(1)

                    fig = go.Figure()
                    # Enveloppe train
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([ts, ts[::-1]]),
                        y=np.concatenate([trm+trs, (trm-trs)[::-1]]),
                        fill="toself", fillcolor="rgba(231,76,60,0.12)",
                        line=dict(color="rgba(0,0,0,0)"), showlegend=False))
                    fig.add_trace(go.Scatter(
                        x=ts, y=trm, mode="lines+markers", name="Entraînement",
                        line=dict(color="#E74C3C", width=2.5), marker=dict(size=8)))
                    # Enveloppe val
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([ts, ts[::-1]]),
                        y=np.concatenate([tem+tes, (tem-tes)[::-1]]),
                        fill="toself", fillcolor="rgba(39,174,96,0.12)",
                        line=dict(color="rgba(0,0,0,0)"), showlegend=False))
                    fig.add_trace(go.Scatter(
                        x=ts, y=tem, mode="lines+markers", name="Validation",
                        line=dict(color="#27AE60", width=2.5), marker=dict(size=8)))

                    fig.update_layout(
                        title=f"<b>{metric_label} — {', '.join(map(str,sorted(A())))} | {', '.join(P())}</b>",
                        xaxis_title="Observations entraînement",
                        yaxis_title=metric_label,
                        height=420, plot_bgcolor="white",
                        legend=dict(x=0.75, y=0.05),
                        yaxis=dict(range=[0, 1.05])
                    )
                    fig.add_annotation(
                        text=f"Val. finale : {tem[-1]:.3f} ± {tes[-1]:.3f}",
                        x=ts[-1], y=tem[-1], showarrow=True, arrowhead=2,
                        ax=40, ay=-30, bgcolor="white",
                        bordercolor="#27AE60", borderwidth=1)
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("📋 Tableau des valeurs"):
                        st.dataframe(pd.DataFrame({
                            "Taille":                   ts,
                            f"{metric_label} Train moy": trm.round(4),
                            f"{metric_label} Train std": trs.round(4),
                            f"{metric_label} Val moy":   tem.round(4),
                            f"{metric_label} Val std":   tes.round(4),
                        }), use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Erreur : {e}")