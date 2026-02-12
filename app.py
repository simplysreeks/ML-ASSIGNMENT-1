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

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="ML Classification App", layout="wide")

st.title("Machine Learning Classification - Assignment 2")
st.write("Breast Cancer Classification using Multiple ML Models")

# ----------------------------
# Load Default Dataset
# ----------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test_default, y_train, y_test_default = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Load Saved Models
# ----------------------------
model_dict = {
    "Logistic Regression": joblib.load("model/logistic_reg.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl"),
}

scaler = joblib.load("model/scaler.pkl")

# ----------------------------
# Sidebar - Upload CSV
# ----------------------------
st.sidebar.header("Upload Test Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)

    if "target" in test_data.columns:
        y_test = test_data["target"]
        X_test = test_data.drop("target", axis=1)
    else:
        st.error("Uploaded CSV must contain a 'target' column.")
        st.stop()
else:
    X_test = X_test_default
    y_test = y_test_default

# Apply scaling
X_test = scaler.transform(X_test)

# ----------------------------
# Model Selection
# ----------------------------
selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(model_dict.keys())
)

model = model_dict[selected_model_name]

# ----------------------------
# Model Evaluation
# ----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

# ----------------------------
# Display Metrics
# ----------------------------
st.subheader("Evaluation Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", round(accuracy, 4))
col2.metric("AUC Score", round(auc, 4))
col3.metric("Precision", round(precision, 4))

col4, col5, col6 = st.columns(3)
col4.metric("Recall", round(recall, 4))
col5.metric("F1 Score", round(f1, 4))
col6.metric("MCC Score", round(mcc, 4))

# ----------------------------
# Confusion Matrix
# ----------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig)
