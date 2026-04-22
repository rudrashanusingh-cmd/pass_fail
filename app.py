import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Pass/Fail Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — dark academia animated theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:      #0e0f1a;
    --card:    #161728;
    --border:  #2a2d4a;
    --accent:  #7c6af7;
    --accent2: #f7c86a;
    --pass:    #3ee8a4;
    --fail:    #f76a6a;
    --text:    #e8e6f0;
    --muted:   #7a7a9a;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
h1,h2,h3,h4 { font-family: 'Playfair Display', serif !important; }

section[data-testid="stSidebar"] {
    background: #10111f !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a1040, #0e1a30, #1a1040);
    background-size: 300% 300%;
    animation: gradientShift 6s ease infinite;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    border: 1px solid var(--border);
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 50% 50%, rgba(124,106,247,0.08) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes pulse {
    0%, 100% { transform: scale(1);   opacity: 0.5; }
    50%       { transform: scale(1.1); opacity: 1; }
}
.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #f7c86a, #3ee8a4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
    animation: fadeSlideDown 0.8s ease both;
}
.hero-sub {
    color: var(--muted) !important;
    font-size: 1rem;
    animation: fadeSlideDown 0.8s 0.2s ease both;
}
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-18px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Cards ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    animation: fadeIn 0.5s ease both;
    transition: border-color 0.3s;
}
.card:hover { border-color: var(--accent); }

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Metric cards ── */
.metric-row  { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 130px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    animation: fadeIn 0.5s ease both;
    transition: transform 0.2s, border-color 0.3s;
}
.metric-card:hover { transform: translateY(-3px); border-color: var(--accent); }
.metric-val { font-size: 2rem; font-weight: 700; font-family: 'Playfair Display', serif; color: var(--accent2); }
.metric-lbl { font-size: 0.8rem; color: var(--muted); margin-top: 0.2rem; }

/* ── Result boxes ── */
.result-pass {
    background: rgba(62,232,164,0.12);
    border: 2px solid var(--pass);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    animation: popIn 0.4s cubic-bezier(0.175,0.885,0.32,1.275) both;
}
.result-fail {
    background: rgba(247,106,106,0.12);
    border: 2px solid var(--fail);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    animation: popIn 0.4s cubic-bezier(0.175,0.885,0.32,1.275) both;
}
@keyframes popIn {
    from { opacity: 0; transform: scale(0.85); }
    to   { opacity: 1; transform: scale(1); }
}
.result-emoji { font-size: 3.5rem; display: block; margin-bottom: 0.4rem; }
.result-label { font-size: 2rem; font-weight: 700; font-family: 'Playfair Display', serif; }
.result-conf  { font-size: 0.95rem; color: var(--muted); margin-top: 0.3rem; }

/* ── Section headers ── */
.section-hdr {
    font-size: 1.05rem;
    font-weight: 500;
    color: var(--accent2);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 1.8rem 0 0.8rem;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Shimmer bar ── */
.shimmer {
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--border) 25%, var(--accent) 50%, var(--border) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    margin-bottom: 1rem;
}
@keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ── Streamlit widget overrides ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #9f8ff5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.65rem 2rem !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 4px 20px rgba(124,106,247,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124,106,247,0.5) !important;
}
.stSelectbox > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent2) !important;
    border-bottom: 2px solid var(--accent2) !important;
}
div.stMarkdown p { color: var(--text) !important; }
.stRadio label    { color: var(--text) !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TRAIN ALL MODELS (cached)
# ─────────────────────────────────────────────
DATA_PATH = "pass_fail_dataset_extended.csv"

@st.cache_resource(show_spinner=False)
def train_all_models():
    df = pd.read_csv(DATA_PATH)
    X = df[["study_hours", "attendance", "previous_score"]]
    y = df["pass"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model_defs = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
    }
    trained = {}
    for name, clf in model_defs.items():
        clf.fit(X_train_sc, y_train)
        y_pred = clf.predict(X_test_sc)
        trained[name] = {
            "model":    clf,
            "accuracy": accuracy_score(y_test, y_pred),
            "report":   classification_report(y_test, y_pred,
                            target_names=["Fail","Pass"], output_dict=True),
            "cm":       confusion_matrix(y_test, y_pred),
        }
    return scaler, trained, df

st.markdown('<div class="shimmer"></div>', unsafe_allow_html=True)
scaler, trained_models, df = train_all_models()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 Student Predictor")
    st.markdown("<hr style='border-color:#2a2d4a'>", unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["🔮 Predict", "📊 Compare Models", "📁 Dataset Explorer"],
        label_visibility="collapsed"
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Show all model accuracies in sidebar
    st.markdown("<div style='font-size:0.82rem;color:#7a7a9a;margin-bottom:0.4rem'>Model Accuracies</div>",
                unsafe_allow_html=True)
    icons  = ["🟣","🟡","🟢"]
    for ico, (name, info) in zip(icons, trained_models.items()):
        short = name.replace("Logistic Regression","LR")
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;font-size:0.82rem;"
            f"padding:0.35rem 0;border-bottom:1px solid #2a2d4a'>"
            f"<span>{ico} {short}</span>"
            f"<b style='color:#f7c86a'>{info['accuracy']*100:.1f}%</b></div>",
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">🎓 Student Pass / Fail Predictor</p>
    <p class="hero-sub">Logistic Regression &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp; Decision Tree</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════
if page == "🔮 Predict":

    col_input, col_result = st.columns([1.1, 1], gap="large")

    with col_input:
        st.markdown('<p class="section-hdr">Student Details</p>', unsafe_allow_html=True)

        model_choice = st.selectbox(
            "🤖 Choose Model",
            list(trained_models.keys()),
        )

        study_hours    = st.slider("📖 Study Hours / day", 0.0, 10.0, 5.0, 0.1)
        attendance     = st.slider("🏫 Attendance %",       50,  99,  75)
        previous_score = st.slider("📝 Previous Score",     40,  99,  65)

        predict_btn = st.button("✨ Predict Now", use_container_width=True)

    with col_result:
        st.markdown('<p class="section-hdr">Prediction Result</p>', unsafe_allow_html=True)

        if predict_btn:
            inp    = np.array([[study_hours, attendance, previous_score]])
            inp_sc = scaler.transform(inp)
            clf    = trained_models[model_choice]["model"]
            pred   = clf.predict(inp_sc)[0]
            proba  = clf.predict_proba(inp_sc)[0]

            with st.spinner("Analysing…"):
                time.sleep(0.4)

            if pred == 1:
                st.markdown(f"""
                <div class="result-pass">
                    <span class="result-emoji">🏆</span>
                    <div class="result-label" style="color:#3ee8a4">PASS</div>
                    <div class="result-conf">Confidence: {proba[1]*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-fail">
                    <span class="result-emoji">📉</span>
                    <div class="result-label" style="color:#f76a6a">FAIL</div>
                    <div class="result-conf">Confidence: {proba[0]*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability bar
            fig, ax = plt.subplots(figsize=(5, 1.8))
            fig.patch.set_facecolor("#161728")
            ax.set_facecolor("#161728")
            bars = ax.barh(["Fail","Pass"], [proba[0], proba[1]],
                           color=["#f76a6a","#3ee8a4"], height=0.5, edgecolor="none")
            ax.set_xlim(0, 1)
            ax.tick_params(colors="#7a7a9a", labelsize=9)
            for spine in ax.spines.values(): spine.set_visible(False)
            for bar, val in zip(bars, [proba[0], proba[1]]):
                ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                        f"{val:.1%}", va="center", color="#e8e6f0", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            acc = trained_models[model_choice]["accuracy"]
            st.markdown(
                f"<div style='text-align:center;font-size:0.82rem;color:#7a7a9a'>"
                f"Model accuracy on test set: <b style='color:#f7c86a'>{acc*100:.1f}%</b></div>",
                unsafe_allow_html=True
            )

            # All-model comparison for same input
            st.markdown('<p class="section-hdr" style="margin-top:1.8rem">All Models on This Input</p>',
                        unsafe_allow_html=True)
            rows = []
            for mname, minfo in trained_models.items():
                p = minfo["model"].predict_proba(inp_sc)[0]
                pred_lbl = "✅ Pass" if minfo["model"].predict(inp_sc)[0] == 1 else "❌ Fail"
                rows.append({"Model": mname, "Prediction": pred_lbl,
                             "Pass %": f"{p[1]*100:.1f}%", "Fail %": f"{p[0]*100:.1f}%"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:3rem 1rem;min-height:240px;
                 display:flex;flex-direction:column;align-items:center;justify-content:center">
                <div style="font-size:3rem;margin-bottom:1rem">🎯</div>
                <div style="color:#7a7a9a">
                    Adjust the sliders and hit <b style="color:#a78bfa">Predict Now</b>
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE 2 — COMPARE MODELS
# ══════════════════════════════════════════════
elif page == "📊 Compare Models":

    names      = list(trained_models.keys())
    accs       = [trained_models[n]["accuracy"] for n in names]
    bar_colors = ["#7c6af7", "#f7c86a", "#3ee8a4"]

    st.markdown('<p class="section-hdr">Accuracy Comparison</p>', unsafe_allow_html=True)

    # Metric cards
    st.markdown('<div class="metric-row">', unsafe_allow_html=True)
    for n, a, c in zip(["LR","Random Forest","Decision Tree"], accs, bar_colors):
        st.markdown(
            f'<div class="metric-card"><div class="metric-val" style="color:{c}">{a*100:.1f}%</div>'
            f'<div class="metric-lbl">{n}</div></div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#161728")
    ax.set_facecolor("#161728")
    bars = ax.bar(names, accs, color=bar_colors, width=0.42, edgecolor="none", zorder=3)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("Accuracy", color="#7a7a9a", fontsize=9)
    ax.tick_params(colors="#7a7a9a", labelsize=9)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.yaxis.grid(True, color="#2a2d4a", zorder=0)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
                f"{val:.3f}", ha="center", color="#e8e6f0", fontsize=10, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Per-model tabs
    st.markdown('<p class="section-hdr">Per-Model Details</p>', unsafe_allow_html=True)
    tabs = st.tabs(names)

    for tab, name in zip(tabs, names):
        with tab:
            info = trained_models[name]
            c1, c2 = st.columns(2, gap="medium")

            with c1:
                st.markdown("**Confusion Matrix**")
                fig2, ax2 = plt.subplots(figsize=(3.5, 3))
                fig2.patch.set_facecolor("#161728")
                ax2.set_facecolor("#161728")
                sns.heatmap(info["cm"], annot=True, fmt="d",
                            cmap=sns.light_palette("#7c6af7", as_cmap=True),
                            xticklabels=["Fail","Pass"], yticklabels=["Fail","Pass"],
                            ax=ax2, linewidths=1, linecolor="#10111f",
                            annot_kws={"size":13,"color":"white"})
                ax2.tick_params(colors="#7a7a9a", labelsize=9)
                ax2.set_xlabel("Predicted", color="#7a7a9a", fontsize=9)
                ax2.set_ylabel("Actual",    color="#7a7a9a", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

            with c2:
                st.markdown("**Classification Report**")
                rdf = pd.DataFrame(info["report"]).transpose().round(3)
                st.dataframe(rdf, use_container_width=True)

            st.markdown("**Feature Importance**")
            features = ["Study Hours", "Attendance", "Prev. Score"]
            if name == "Logistic Regression":
                vals   = info["model"].coef_[0]
                ylabel = "Coefficient"
                fcolors = ["#3ee8a4" if v > 0 else "#f76a6a" for v in vals]
            else:
                vals    = info["model"].feature_importances_
                ylabel  = "Importance"
                fcolors = ["#7c6af7","#f7c86a","#3ee8a4"]

            fig3, ax3 = plt.subplots(figsize=(6, 2.4))
            fig3.patch.set_facecolor("#161728")
            ax3.set_facecolor("#161728")
            ax3.bar(features, vals, color=fcolors, edgecolor="none", width=0.45)
            ax3.axhline(0, color="#2a2d4a", linewidth=0.8)
            ax3.tick_params(colors="#7a7a9a", labelsize=9)
            ax3.set_ylabel(ylabel, color="#7a7a9a", fontsize=9)
            for spine in ax3.spines.values(): spine.set_visible(False)
            ax3.yaxis.grid(True, color="#2a2d4a", zorder=0)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

# ══════════════════════════════════════════════
# PAGE 3 — DATASET EXPLORER
# ══════════════════════════════════════════════
elif page == "📁 Dataset Explorer":

    total  = len(df)
    passed = int(df["pass"].sum())
    failed = total - passed

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-val">{total:,}</div>
            <div class="metric-lbl">Total Records</div>
        </div>
        <div class="metric-card">
            <div class="metric-val" style="color:#3ee8a4">{passed:,}</div>
            <div class="metric-lbl">Passed ✅</div>
        </div>
        <div class="metric-card">
            <div class="metric-val" style="color:#f76a6a">{failed:,}</div>
            <div class="metric-lbl">Failed ❌</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">{passed/total*100:.1f}%</div>
            <div class="metric-lbl">Pass Rate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">Sample Data</p>', unsafe_allow_html=True)
    st.dataframe(df.head(15), use_container_width=True)

    st.markdown('<p class="section-hdr">Feature Distributions</p>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    fig.patch.set_facecolor("#161728")
    cols_info = [
        ("study_hours",    "Study Hours",    "#7c6af7"),
        ("attendance",     "Attendance (%)", "#f7c86a"),
        ("previous_score", "Previous Score", "#3ee8a4"),
    ]
    for ax, (col, title, color) in zip(axes, cols_info):
        ax.set_facecolor("#161728")
        df[df["pass"]==0][col].hist(ax=ax, bins=25, alpha=0.6, color="#f76a6a", label="Fail", edgecolor="none")
        df[df["pass"]==1][col].hist(ax=ax, bins=25, alpha=0.6, color=color,    label="Pass", edgecolor="none")
        ax.set_title(title, color="#e8e6f0", fontsize=10, pad=8)
        ax.tick_params(colors="#7a7a9a", labelsize=8)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.yaxis.grid(True, color="#2a2d4a", zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, facecolor="#1c1e35", edgecolor="#2a2d4a",
                  labelcolor="#e8e6f0", framealpha=0.9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<p class="section-hdr">Correlation Heatmap</p>', unsafe_allow_html=True)
    fig4, ax4 = plt.subplots(figsize=(5, 3.5))
    fig4.patch.set_facecolor("#161728")
    ax4.set_facecolor("#161728")
    sns.heatmap(df.corr(), annot=True, fmt=".2f",
                cmap=sns.diverging_palette(260, 140, as_cmap=True),
                ax=ax4, linewidths=1, linecolor="#10111f",
                annot_kws={"size":10,"color":"white"})
    ax4.tick_params(colors="#7a7a9a", labelsize=9)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()
