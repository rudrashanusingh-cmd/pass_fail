import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Student Pass/Fail Predictor", page_icon="🎓", layout="centered")

MODEL_PATH  = "model.pkl"
SCALER_PATH = "scaler.pkl"
DATA_PATH   = "pass_fail_dataset_extended.csv"

# ── Train & save model (runs once) ───────────────────────────────────────────
@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH)
        X = df[["study_hours", "attendance", "previous_score"]]
        y = df["pass"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train_sc, y_train)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

    return model, scaler, df


model, scaler, df = load_or_train_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio("Go to", ["🔮 Predict", "📊 Model Info", "📁 Dataset"])

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 – PREDICT
# ════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict":
    st.title("🎓 Student Pass / Fail Predictor")
    st.markdown("Enter the student details below to predict whether the student will **pass** or **fail**.")

    col1, col2 = st.columns(2)
    with col1:
        study_hours    = st.slider("📖 Study Hours (per day)", 0.0, 10.0, 5.0, 0.1)
        attendance     = st.slider("🏫 Attendance (%)", 50, 99, 75)
    with col2:
        previous_score = st.slider("📝 Previous Score", 40, 99, 65)

    st.markdown("---")

    if st.button("Predict", use_container_width=True):
        input_data = np.array([[study_hours, attendance, previous_score]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        if prediction == 1:
            st.success(f"✅ **PASS** — Confidence: {proba[1]*100:.1f}%")
        else:
            st.error(f"❌ **FAIL** — Confidence: {proba[0]*100:.1f}%")

        # Probability bar
        st.markdown("#### Prediction Probability")
        prob_df = pd.DataFrame({"Outcome": ["Fail", "Pass"], "Probability": [proba[0], proba[1]]})
        fig, ax = plt.subplots(figsize=(5, 2))
        colors = ["#e74c3c", "#2ecc71"]
        ax.barh(prob_df["Outcome"], prob_df["Probability"], color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        for i, v in enumerate(prob_df["Probability"]):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center")
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 – MODEL INFO
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Info":
    st.title("📊 Model Performance")

    X = df[["study_hours", "attendance", "previous_score"]]
    y = df["pass"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_sc = scaler.transform(X_test)
    y_pred = model.predict(X_test_sc)

    acc = accuracy_score(y_test, y_pred)
    st.metric("✅ Accuracy", f"{acc*100:.2f}%")

    st.markdown("#### Classification Report")
    report = classification_report(y_test, y_pred, target_names=["Fail", "Pass"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Feature Coefficients")
    coef_df = pd.DataFrame({
        "Feature": ["Study Hours", "Attendance", "Previous Score"],
        "Coefficient": model.coef_[0]
    }).sort_values("Coefficient", ascending=False)
    fig2, ax2 = plt.subplots(figsize=(5, 2.5))
    colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in coef_df["Coefficient"]]
    ax2.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Coefficient Value")
    st.pyplot(fig2)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 – DATASET
# ════════════════════════════════════════════════════════════════════════════
elif page == "📁 Dataset":
    st.title("📁 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", len(df))
    col2.metric("Passed", df["pass"].sum())
    col3.metric("Failed", (df["pass"] == 0).sum())

    st.markdown("#### Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("#### Distribution of Study Hours")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, col, title in zip(axes,
                               ["study_hours", "attendance", "previous_score"],
                               ["Study Hours", "Attendance (%)", "Previous Score"]):
        df[df["pass"] == 0][col].hist(ax=ax, alpha=0.6, label="Fail", color="#e74c3c", bins=20)
        df[df["pass"] == 1][col].hist(ax=ax, alpha=0.6, label="Pass", color="#2ecc71", bins=20)
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
