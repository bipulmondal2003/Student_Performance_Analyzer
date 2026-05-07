"""
app.py  —  AI-Powered Student Performance Analyzer
Run with:  streamlit run app.py
"""

import os
import sys

# ── Ensure src/ is importable ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import streamlit as st
import numpy as np
import pandas as pd

# ── Internal modules ─────────────────────────────────────────────────────────
from data_processing import load_data, preprocess_data, get_feature_names
from train          import train_all_models, save_model, load_model
from evaluate       import (
    get_confusion_matrix_fig,
    get_cross_val_fig,
    get_feature_importance_fig,
    get_cv_scores,
)
from predict   import predict_student
from suggest   import generate_suggestions
from visualize import (
    scatter_study_vs_result,
    scatter_attendance_vs_result,
    bar_model_accuracy,
    histogram_features,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Student Performance Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ────────────────────────────────────────────────────────
css_path = os.path.join(BASE_DIR, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "trained":      False,
        "results":      None,
        "scaler":       None,
        "df":           None,
        "X_train":      None,
        "X_test":       None,
        "y_train":      None,
        "y_test":       None,
        "X_full":       None,
        "y_full":       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def _load_and_train():
    """Load data, preprocess, train models — cached in session state."""
    df = load_data()
    st.session_state["df"] = df

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    st.session_state.update({
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "scaler":  scaler,
        "X_full":  scaler.transform(df[get_feature_names()].values),
        "y_full":  df["result"].values,
    })

    results = train_all_models(X_train, X_test, y_train, y_test)
    st.session_state["results"] = results
    st.session_state["trained"] = True

    # Persist best model to disk
    save_model(results["best_model"], scaler)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 Student Analyzer")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🧠 Train Model", "🔮 Prediction", "📊 Visualization", "📈 Model Performance"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Built with Python · Streamlit · scikit-learn")

# ═══════════════════════════════════════════════════════════════════════════════
# ① HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# 🎓 AI Student Performance Analyzer")
    st.markdown("### Predict academic outcomes & get personalised improvement tips")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        This application uses **Machine Learning** to predict whether a student will
        **PASS** or **FAIL** based on three key academic indicators.
        It compares two classification models, evaluates their performance, and
        provides actionable AI-driven suggestions to help students improve.
        """)

        st.markdown("#### 🚀 Features")
        features = [
            ("🧠", "Train Model",        "Compare Logistic Regression vs Decision Tree"),
            ("🔮", "Prediction",         "Instant PASS / FAIL prediction with confidence score"),
            ("💡", "Smart Suggestions",  "Personalised improvement tips based on your inputs"),
            ("📊", "Visualizations",     "Interactive scatter plots and accuracy charts"),
            ("📈", "Model Performance",  "Confusion matrix, cross-validation & feature importance"),
        ]
        for icon, title, desc in features:
            st.markdown(
                f"""<div class="info-card">
                    <strong>{icon} {title}</strong><br>
                    <span style="color:#bdc3c7">{desc}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### 📋 Dataset Overview")
        st.markdown("""
        | Feature | Range |
        |---|---|
        | Study Hours | 0 – 10 hrs |
        | Attendance | 0 – 100 % |
        | Previous Marks | 0 – 100 |
        | Result | PASS / FAIL |
        """)
        st.markdown("#### 🤖 Models Used")
        st.markdown("""
        - Logistic Regression
        - Decision Tree
        """)
        st.info("👈 Use the sidebar to navigate between sections.")

    st.markdown("---")
    st.markdown("#### ⚡ Quick Start")
    st.markdown("1. Go to **Train Model** and click *Train Models*")
    st.markdown("2. Head to **Prediction** to input student data")
    st.markdown("3. Explore **Visualization** and **Model Performance**")


# ═══════════════════════════════════════════════════════════════════════════════
# ② TRAIN MODEL PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Train Model":
    st.markdown("# 🧠 Train Models")
    st.markdown("Click the button below to load the dataset and train both classifiers.")
    st.markdown("---")

    if st.button("🚀 Train Models", use_container_width=False):
        with st.spinner("Loading data and training models …"):
            try:
                results = _load_and_train()
                st.success("✅ Models trained successfully!")
            except FileNotFoundError as e:
                st.error(f"❌ {e}")
                st.stop()

    if st.session_state["trained"]:
        results = st.session_state["results"]
        st.markdown("### 📊 Model Accuracies")

        col1, col2, col3 = st.columns(3)

        lr_acc = results["Logistic Regression"]["accuracy"]
        dt_acc = results["Decision Tree"]["accuracy"]
        best   = results["best_model_name"]

        with col1:
            st.metric("🔵 Logistic Regression", f"{lr_acc*100:.1f}%")

        with col2:
            st.metric("🟣 Decision Tree", f"{dt_acc*100:.1f}%")

        with col3:
            st.metric("🏆 Best Model", best)

        st.markdown("---")
        st.markdown("### 📉 Accuracy Comparison Chart")
        fig = bar_model_accuracy(results)
        st.pyplot(fig, use_container_width=True)

        st.markdown("---")
        diff = abs(lr_acc - dt_acc) * 100
        if diff < 2:
            st.info("💡 Both models perform similarly on this dataset.")
        elif best == "Logistic Regression":
            st.info("💡 Logistic Regression edges out Decision Tree — likely a linearly separable dataset.")
        else:
            st.info("💡 Decision Tree outperforms Logistic Regression — feature interactions are non-linear.")
    else:
        st.info("⬆️ Click **Train Models** to get started.")


# ═══════════════════════════════════════════════════════════════════════════════
# ③ PREDICTION PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    st.markdown("# 🔮 Student Performance Prediction")
    st.markdown("Adjust the sliders and click **Predict Result**.")
    st.markdown("---")

    # ── Ensure model is available ────────────────────────────────────────────
    if not st.session_state["trained"]:
        # Try loading a previously saved model
        model, scaler = load_model()
        if model is not None:
            st.session_state["scaler"] = scaler
            # Rebuild minimal results for prediction
            st.session_state["trained"]  = True
            st.session_state["results"]  = {
                "best_model": model,
                "best_model_name": "Saved Model",
                "Logistic Regression": {"model": model, "accuracy": 0},
                "Decision Tree":       {"model": model, "accuracy": 0},
            }
            st.info("✅ Loaded previously trained model from disk.")
        else:
            st.warning("⚠️ Please train the models first on the **Train Model** page.")
            st.stop()

    col_inp, col_out = st.columns([1, 1])

    with col_inp:
        st.markdown("#### 🎛️ Student Input")
        study_hours    = st.slider("📚 Study Hours / Day",  0.0, 10.0, 4.0, 0.5)
        attendance     = st.slider("🏫 Attendance (%)",     0.0, 100.0, 75.0, 1.0)
        previous_marks = st.slider("📝 Previous Marks",     0.0, 100.0, 60.0, 1.0)

        predict_btn = st.button("🔍 Predict Result", use_container_width=True)

    with col_out:
        st.markdown("#### 📋 Prediction Result")

        if predict_btn:
            model  = st.session_state["results"]["best_model"]
            scaler = st.session_state["scaler"]
            label, confidence, proba_pass = predict_student(
                model, scaler, study_hours, attendance, previous_marks
            )

            # ── Result badge ─────────────────────────────────────────────────
            badge_class = "pass-badge" if label == "PASS" else "fail-badge"
            st.markdown(
                f'<div style="text-align:center;margin:20px 0">'
                f'<span class="{badge_class}">{label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Confidence ───────────────────────────────────────────────────
            st.markdown(f"**Confidence:** `{confidence*100:.1f}%`")
            st.progress(int(confidence * 100))

            # ── Probability bar ───────────────────────────────────────────────
            st.markdown("**PASS Probability**")
            col_a, col_b = st.columns(2)
            col_a.metric("🟢 PASS", f"{proba_pass*100:.1f}%")
            col_b.metric("🔴 FAIL", f"{(1-proba_pass)*100:.1f}%")

            st.markdown("---")

            # ── Suggestions ──────────────────────────────────────────────────
            st.markdown("#### 💡 AI Improvement Suggestions")
            suggestions = generate_suggestions(study_hours, attendance, previous_marks)

            level_colors = {
                "success": ("#2ecc71", "✅"),
                "warning": ("#f39c12", "⚠️"),
                "error":   ("#e74c3c", "🚨"),
            }
            for s in suggestions:
                color, _ = level_colors.get(s["level"], ("#3498db", "ℹ️"))
                st.markdown(
                    f"""<div style="border-left:4px solid {color};
                                    background:#1a1a2e;
                                    border-radius:6px;
                                    padding:12px 16px;
                                    margin-bottom:10px">
                        <strong style="color:{color}">{s['icon']} {s['title']}</strong><br>
                        <span style="color:#bdc3c7">{s['detail']}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.info("👈 Set the sliders and press **Predict Result**.")


# ═══════════════════════════════════════════════════════════════════════════════
# ④ VISUALIZATION PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Visualization":
    st.markdown("# 📊 Data Visualizations")
    st.markdown("---")

    if not st.session_state["trained"] or st.session_state["df"] is None:
        # Auto-train silently so visualizations work without manual step
        with st.spinner("Loading dataset …"):
            try:
                _load_and_train()
            except FileNotFoundError as e:
                st.error(f"❌ {e}")
                st.stop()

    df = st.session_state["df"]

    # ── Quick stats ──────────────────────────────────────────────────────────
    st.markdown("### 📋 Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students",  len(df))
    c2.metric("PASS",            int(df["result"].sum()))
    c3.metric("FAIL",            int((df["result"] == 0).sum()))
    c4.metric("Pass Rate",       f"{df['result'].mean()*100:.1f}%")

    st.markdown("---")

    # ── Plot 1: Study Hours ──────────────────────────────────────────────────
    st.markdown("### 1️⃣  Study Hours vs Previous Marks")
    st.pyplot(scatter_study_vs_result(df), use_container_width=True)
    st.caption("Green circles = PASS · Red crosses = FAIL")

    st.markdown("---")

    # ── Plot 2: Attendance ───────────────────────────────────────────────────
    st.markdown("### 2️⃣  Attendance vs Previous Marks")
    st.pyplot(scatter_attendance_vs_result(df), use_container_width=True)
    st.caption("Orange dashed line marks the 70% attendance threshold.")

    st.markdown("---")

    # ── Plot 3: Model Accuracy ───────────────────────────────────────────────
    st.markdown("### 3️⃣  Model Accuracy Comparison")
    results = st.session_state["results"]
    st.pyplot(bar_model_accuracy(results), use_container_width=True)
    st.caption("Gold border highlights the best-performing model.")

    st.markdown("---")

    # ── Plot 4: Feature distributions ───────────────────────────────────────
    st.markdown("### 4️⃣  Feature Distributions by Outcome")
    st.pyplot(histogram_features(df), use_container_width=True)

    st.markdown("---")

    # ── Raw data ─────────────────────────────────────────────────────────────
    with st.expander("🗃️ View Raw Dataset"):
        st.dataframe(df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ⑤ MODEL PERFORMANCE PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown("# 📈 Model Performance")
    st.markdown("---")

    if not st.session_state["trained"]:
        with st.spinner("Training models for evaluation …"):
            try:
                _load_and_train()
            except FileNotFoundError as e:
                st.error(f"❌ {e}")
                st.stop()

    results = st.session_state["results"]
    X_test  = st.session_state["X_test"]
    y_test  = st.session_state["y_test"]
    X_full  = st.session_state["X_full"]
    y_full  = st.session_state["y_full"]

    # ── Model selector ───────────────────────────────────────────────────────
    model_choice = st.selectbox("Select model to inspect", ["Logistic Regression", "Decision Tree"])
    selected_model = results[model_choice]["model"]

    st.markdown("---")

    # ── Confusion Matrix ─────────────────────────────────────────────────────
    st.markdown("### 🔲 Confusion Matrix")
    fig_cm = get_confusion_matrix_fig(selected_model, X_test, y_test, model_choice)
    st.pyplot(fig_cm, use_container_width=False)

    st.markdown("---")

    # ── Cross-Validation ─────────────────────────────────────────────────────
    st.markdown("### 🔁 Cross-Validation (cv = 5)")
    fig_cv = get_cross_val_fig(results, X_full, y_full)
    st.pyplot(fig_cv, use_container_width=True)

    # Individual scores table
    with st.expander(f"📄 Fold-by-fold scores for {model_choice}"):
        cv_scores = get_cv_scores(selected_model, X_full, y_full)
        cv_df = pd.DataFrame({
            "Fold":     [f"Fold {i+1}" for i in range(len(cv_scores))],
            "Accuracy": [f"{s*100:.2f}%" for s in cv_scores],
        })
        st.dataframe(cv_df, use_container_width=True)
        st.metric("Mean Accuracy", f"{cv_scores.mean()*100:.2f}%")
        st.metric("Std Dev",       f"± {cv_scores.std()*100:.2f}%")

    st.markdown("---")

    # ── Feature Importance (Decision Tree only) ──────────────────────────────
    st.markdown("### 🌳 Feature Importance (Decision Tree)")
    dt_model = results["Decision Tree"]["model"]
    fig_fi = get_feature_importance_fig(dt_model, get_feature_names())
    st.pyplot(fig_fi, use_container_width=True)
    st.caption("Higher bars = feature has stronger influence on the prediction.")

    st.markdown("---")

    # ── Summary Card ─────────────────────────────────────────────────────────
    st.markdown("### 🏆 Overall Summary")
    best = results["best_model_name"]
    best_acc = results[best]["accuracy"]
    st.success(f"**Best Model:** {best}  |  **Test Accuracy:** {best_acc*100:.1f}%")
