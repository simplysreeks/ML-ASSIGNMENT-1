import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="ML Classification Dashboard",
    layout="wide",
    page_icon="📊"
)

# ----------------------------------------
# Custom Styling
# ----------------------------------------
st.markdown("""
<style>
.metric-container {
    padding: 15px;
    border-radius: 10px;
    background-color: #f8f9fa;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# Header
# ----------------------------------------
st.title("📊 Machine Learning Classification Dashboard")
st.markdown("### Breast Cancer Prediction using Multiple ML Models")
st.markdown("---")

# ----------------------------------------
# Load Dataset
# ----------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test_default, y_train, y_test_default = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------
# Load Models
# ----------------------------------------
model_dict = {
    "Logistic Regression": joblib.load("model/logistic_reg.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl"),
}

scaler = joblib.load("model/scaler.pkl")

# ----------------------------------------
# Sidebar
# ----------------------------------------
st.sidebar.title("⚙️ Configuration")

st.sidebar.subheader("Upload Test Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)

    if "target" in test_data.columns:
        y_test = test_data["target"]
        X_test = test_data.drop("target", axis=1)
    else:
        st.sidebar.error("CSV must contain 'target' column.")
        st.stop()
else:
    X_test = X_test_default
    y_test = y_test_default

# Apply scaling
X_test = scaler.transform(X_test)

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(model_dict.keys())
)

model = model_dict[selected_model_name]

# ----------------------------------------
# Model Description
# ----------------------------------------
st.markdown(f"### 🔍 Selected Model: {selected_model_name}")

model_descriptions = {
    "Logistic Regression": "A linear model suitable for binary classification problems.",
    "Decision Tree": "A tree-based model that splits data using decision rules.",
    "KNN": "Classifies based on the majority class among nearest neighbors.",
    "Naive Bayes": "Probabilistic classifier based on Bayes theorem.",
    "Random Forest": "Ensemble of decision trees to improve stability and accuracy.",
    "XGBoost": "Advanced gradient boosting algorithm with strong performance."
}

st.info(model_descriptions[selected_model_name])

# ----------------------------------------
# Evaluation
# ----------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

st.markdown("## 📈 Evaluation Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("AUC Score", f"{auc:.4f}")
col3.metric("Precision", f"{precision:.4f}")

col4, col5, col6 = st.columns(3)
col4.metric("Recall", f"{recall:.4f}")
col5.metric("F1 Score", f"{f1:.4f}")
col6.metric("MCC Score", f"{mcc:.4f}")

# ----------------------------------------
# Confusion Matrix
# ----------------------------------------
st.markdown("## 📊 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

st.pyplot(fig)

# ----------------------------------------
# Footer
# ----------------------------------------
st.markdown("---")
st.caption("Developed for Machine Learning Assignment 2 | BITS Pilani WILP")
