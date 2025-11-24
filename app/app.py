# app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
from streamlit_shap import st_shap

# ------------------------------------------------------------
# 0. PATH SETUP
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# 1. DEBUG PANEL
# ------------------------------------------------------------
with st.expander("🔍 DEBUG INFO", expanded=True):
    st.write("**BASE_DIR** →", BASE_DIR)
    model_dir = os.path.join(BASE_DIR, "..", "model")
    csv_path = os.path.join(BASE_DIR, "..", "data", "preprocessed", "X_train_smote.csv")
    st.write("**Model folder:**", model_dir)
    st.write("**Gradient Boosting model exists?**", os.path.exists(os.path.join(model_dir, "gradient_boosting_model.pkl")))
    st.write("**Random Forest model exists?**", os.path.exists(os.path.join(model_dir, "random_forest_model.pkl")))
    st.write("**Logistic Regression model exists?**", os.path.exists(os.path.join(model_dir, "logistic_regression_model.pkl")))
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
# 3. LOAD SHAP EXPLAINERS (OPTIONAL)
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
csv_path = os.path.join(BASE_DIR, "..", "data", "preprocessed", "X_train_smote.csv")
try:
    X_train_smote = pd.read_csv(csv_path)
except Exception as e:
    st.error(f"Could not load X_train_smote.csv: {e}")
    X_train_smote = pd.DataFrame()

# ------------------------------------------------------------
# 5. PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")

# ------------------------------------------------------------
# 6. INPUT FEATURES (Sidebar Sliders)
# ------------------------------------------------------------
st.sidebar.header("Transaction Features")
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
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox("Choose model", ("Gradient Boosting", "Random Forest", "Logistic Regression"))

model = {"Gradient Boosting": gb_model, "Random Forest": rf_model, "Logistic Regression": lr_model}[model_choice]
explainer = {"Gradient Boosting": gb_explainer, "Random Forest": rf_explainer, "Logistic Regression": lr_explainer}[model_choice]

# ------------------------------------------------------------
# 8. UPLOAD & TEST DATA
# ------------------------------------------------------------
st.sidebar.header("📂 Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file for testing", type=["csv"])

uploaded_df = None
if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(uploaded_df.head())

        st.write(f"✅ Uploaded dataset contains `{uploaded_df.shape[0]}` rows and `{uploaded_df.shape[1]}` columns.")
        missing_cols = [col for col in X_train_smote.columns if col not in uploaded_df.columns]
        if missing_cols:
            st.warning(f"⚠️ Missing columns: {missing_cols}")
        else:
            st.success("✅ All required columns are present. Ready to test!")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")

# ------------------------------------------------------------
# 9. PREDICTION + SHAP
# ------------------------------------------------------------
if model and (not input_df.empty or uploaded_df is not None):
    # Determine input source
    data_to_predict = uploaded_df if uploaded_df is not None else input_df

    # Perform prediction
    pred = model.predict(data_to_predict)
    prob = model.predict_proba(data_to_predict)[:, 1]

    st.subheader("🔎 Prediction Results")
    results = data_to_predict.copy()
    results["Prediction"] = pred
    results["Fraud_Probability"] = prob
    st.dataframe(results.head())

    # For single-row inputs, show SHAP explanation
    if results.shape[0] == 1:
        st.subheader(f"🔍 SHAP Explanation — {model_choice}")
        try:
            if model_choice == "Random Forest":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(data_to_predict)
                if isinstance(shap_values, list):
                    shap_val = shap_values[1]
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    shap_val = shap_values[:, :, 1]
                else:
                    shap_val = shap_values
                expected_value = explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[1]
            else:
                shap_values = explainer(data_to_predict)
                shap_val = shap_values.values
                expected_value = explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[0]

            # --- Interactive Force Plot ---
            st.write("**Interactive Force Plot**")
            force_plot = shap.force_plot(expected_value, shap_val, data_to_predict, show=False)
            st_shap(force_plot)

            # --- Global Feature Importance ---
            st.write("**Global Feature Importance**")
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.summary_plot(shap_val, data_to_predict, plot_type="bar", show=False)
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.warning(f"SHAP computation failed: {e}")

    # ---- Save ----
    if st.sidebar.button("💾 Save Predictions"):
        save_dir = os.path.join(BASE_DIR, "data")
        os.makedirs(save_dir, exist_ok=True)
        csv_path_out = os.path.join(save_dir, "predictions.csv")
        header = not os.path.exists(csv_path_out)
        results.to_csv(csv_path_out, mode="a", header=header, index=False)
        st.success("✅ Predictions saved successfully!")

else:
    st.info("⚙️ Waiting for models or data input...")
