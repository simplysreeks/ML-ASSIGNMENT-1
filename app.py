import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ----------------------------------------
# Page Config
# ----------------------------------------
st.set_page_config(page_title="ML Dashboard", layout="wide")

st.title("📊 Machine Learning Classification Dashboard")
st.markdown("### Breast Cancer Prediction using Multiple ML Models")
st.markdown("---")

# ----------------------------------------
# Load Dataset
# ----------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Split
X_train, X_test_default, y_train, y_test_default = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_default_scaled = scaler.transform(X_test_default)

# ----------------------------------------
# Sidebar
# ----------------------------------------
st.sidebar.title("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost",
    ]
)

# ----------------------------------------
# Train Model Based on Selection
# ----------------------------------------
model_mapping = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss"),
}

model = model_mapping[selected_model_name]
model.fit(X_train_scaled, y_train)

# ----------------------------------------
# Handle Uploaded CSV
# ----------------------------------------
if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)

    if "target" not in test_data.columns:
        st.sidebar.error("CSV must contain 'target' column.")
        st.stop()

    y_test = test_data["target"]
    X_test = test_data.drop("target", axis=1)

    try:
        X_test = X_test[X.columns]
    except:
        st.sidebar.error("Uploaded CSV columns do not match expected features.")
        st.stop()

    X_test_scaled = scaler.transform(X_test)

else:
    X_test_scaled = X_test_default_scaled
    y_test = y_test_default

# ----------------------------------------
# Predictions
# ----------------------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

# ----------------------------------------
# Display Metrics
# ----------------------------------------
st.markdown(f"## 🔍 Selected Model: {selected_model_name}")
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

st.markdown("---")
st.caption("Developed for ML Assignment 2 | BITS Pilani WILP")
