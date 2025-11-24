# app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------
# 0. PATH SETUP – THIS MUST BE FIRST
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# 1. DEBUG: SHOW PATHS (REMOVE LATER)
# ------------------------------------------------------------
with st.expander("DEBUG: Click to hide", expanded=True):
    st.write("**BASE_DIR** →", BASE_DIR)
    model_dir = os.path.join(BASE_DIR, "..", "model")
    csv_path = os.path.join(BASE_DIR, "..", "preprocessed", "data", "X_train_smote.csv")
    st.write("**Model folder:**", model_dir)
    st.write("**Model exists?**", os.path.exists(os.path.join(model_dir, "gradient_boosting_model.pkl")))
    st.write("**CSV exists?**", os.path.exists(csv_path))

# ------------------------------------------------------------
# 2. LOAD MODELS
# ------------------------------------------------------------
@st.cache_data
def load_models():
    model_dir = os.path.join(BASE_DIR, "..", "model")
    try:
        gb = joblib.load(os.path.join(model_dir, "gradient_boosting_model.pkl"))
        rf = joblib.load(os.path.join(model_dir, "random_forest_model.pkl"))
        lr = joblib.load(os.path.join(model_dir, "logistic_regression_model.pkl"))
        return gb, rf, lr
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

gb_model, rf_model, lr_model = load_models()

# ------------------------------------------------------------
# 3. LOAD SHAP EXPLAINERS
# ------------------------------------------------------------
@st.cache_data
def load_shap_explainers():
    model_dir = os.path.join(BASE_DIR, "..", "model")
    try:
        gb = joblib.load(os.path.join(model_dir, "SHAP_Explainer_Gradient_Boosting.pkl"))
        rf = joblib.load(os.path.join(model_dir, "SHAP_Explainer_Random_Forest.pkl"))
        lr = joblib.load(os.path.join(model_dir, "SHAP_Explainer_Logistic_Regression.pkl"))
        return gb, rf, lr
    except Exception as e:
        st.warning(f"SHAP explainers could not be loaded: {e}")
        return None, None, None

gb_explainer, rf_explainer, lr_explainer = load_shap_explainers()

# ------------------------------------------------------------
# 4. LOAD TRAINING DATA
# ------------------------------------------------------------
csv_path = os.path.join(BASE_DIR, "..", "preprocessed", "data", "X_train_smote.csv")
try:
    X_train_smote = pd.read_csv(csv_path)
except Exception as e:
    st.error(f"Could not load X_train_smote.csv: {e}")
    X_train_smote = pd.DataFrame()

# ------------------------------------------------------------
# 5. PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="Credit Card", layout="wide")
st.title("Credit Card Fraud Detection Dashboard")

# ------------------------------------------------------------
# 6. INPUT FEATURES
# ------------------------------------------------------------
st.sidebar.header("Input Transaction Features")
def user_input_features():
    if X_train_smote.empty:
        st.warning("Training data not loaded — sliders disabled.")
        return pd.DataFrame()
    data = {}
    for col in X_train_smote.columns:
        mn, mx, avg = X_train_smote[col].min(), X_train_smote[col].max(), X_train_smote[col].mean()
        data[col] = st.sidebar.slider(col, float(mn), float(mx), float(avg))
    return pd.DataFrame([data])

input_df = user_input_features()

# ------------------------------------------------------------
# 7. MODEL SELECTION
# ------------------------------------------------------------
st.sidebar.header("Choose Model")
model_choice = st.sidebar.selectbox("Select model", ("Gradient Boosting", "Random Forest", "Logistic Regression"))

model = {"Gradient Boosting": gb_model, "Random Forest": rf_model, "Logistic Regression": lr_model}[model_choice]
explainer = {"Gradient Boosting": gb_explainer, "Random Forest": rf_explainer, "Logistic Regression": lr_explainer}[model_choice]

# ------------------------------------------------------------
# 8. PREDICTION + SHAP
# ------------------------------------------------------------
if model and not input_df.empty:
    pred = int(model.predict(input_df)[0])
    prob = float(model.predict_proba(input_df)[0][1])

    st.subheader("Prediction")
    st.write(f"**Result:** {'Fraudulent' if pred else 'Legitimate'}")
    st.write(f"**Fraud Probability:** {prob:.4f}")

    st.subheader(f"SHAP Explanation – {model_choice}")
    try:
        shap_vals = explainer(input_df)
        html = shap.force_plot(explainer.expected_value, shap_vals.values, input_df, show=False)._repr_html_()
        st.components.v1.html(html, height=300)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_vals.values, input_df, plot_type="bar", show=False)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"SHAP error: {e}")

    if st.sidebar.button("Save Transaction"):
        os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
        out = input_df.copy()
        out["Prediction"] = pred
        out["Fraud_Probability"] = prob
        out.to_csv(os.path.join(BASE_DIR, "data", "predictions.csv"), mode="a", header=not os.path.exists("data/predictions.csv"), index=False)
        st.success("Saved!")
else:
    st.info("Waiting for models and data...")