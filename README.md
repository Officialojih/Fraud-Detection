
# 💳 Credit Card Fraud Detection

A complete end‑to‑end **Machine Learning fraud detection system** built with:
- Python
- Scikit‑learn
- SMOTE
- Gradient Boosting, Random Forest, Logistic Regression
- SHAP Explainability
- Streamlit Web App

This project predicts fraudulent credit card transactions and provides full explainability for each prediction.

---

## 🎬 Demo
[![Dashboard Demo](assets/dashboard_demo.gif) ](https://fraud-detection-f5495m9zigrh233l9hmqeu.streamlit.app) 
*Interactive fraud detection dashboard (Streamlit).*

---

## 🚀 Features
- **Model Training Pipeline** (SMOTE, preprocessing, evaluation)
- **Three Trained Models**
  - Gradient Boosting
  - Random Forest
  - Logistic Regression
- **Confusion Matrix & ROC Curves**
- **SHAP Feature Importance**
- **Streamlit Web App** for interactive fraud prediction
- **Probability Output** for each prediction
- **Automatic saving** of predictions

---

## 📁 Project Structure
```
Fraud-Detection/\
│
├── app/\
│ └── app.py # Streamlit application\
│
├── model/\
│ ├── gradient_boosting_model.pkl\
│ ├── random_forest_model.pkl\
│ ├── logistic_regression_model.pkl\
│ ├── SHAP_Explainer_Gradient_Boosting.pkl\
│ ├── SHAP_Explainer_Random_Forest.pkl\
│ ├── SHAP_Explainer_Logistic_Regression.pkl\
│
├── data/\
│ └── preprocessed/\
│ └── X_train_smote.csv\
│
├── requirements.txt\
└── README.md
```

---

## 🛠️ Installation & Setup
### 1. Clone this repository
```bash
git clone https://github.com/Officialojih/Fraud-Detection.git
cd Fraud-Detection
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app

```bash
streamlit run app/app.py
```

🧠 Machine Learning Workflow

1. Load and preprocess dataset

2. Apply SMOTE oversampling

3. Train 3 ML models

4. Evaluate using:

   * Accuracy

   * Precision

   * Recall

   * F1-score

   * ROC‑AUC

5. Save best models using joblib

6. Generate SHAP explainers

📊 Streamlit App Functions

* Input transaction features with slider ranges

* Choose model (GB / RF / LR)

* Predict whether transaction is fraudulent or legitimate

* Display fraud probability

* Visual SHAP explanations:

  * Force plot

  * Global importance bar chart

* Save predictions to CSV

🖼️ Screenshots / Plots
Confusion Matrix Example
 
SHAP Feature Importance
 
Architecture Diagram
 
👤 Author
James Ojih\
Mechatronics Engineer • Data Scientist • ML Engineer
🔗 LinkedIn: https://www.linkedin.com/in/engr-james-ojih-8838872a6/\
🐙 GitHub: https://github.com/Officialojih
If you need someone eager to learn, improve, and contribute, I’m up for hire.
⭐ Contribute
Contributions are welcome!\
If this project helped you, kindly star ⭐ the repo.
