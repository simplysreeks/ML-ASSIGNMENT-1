import numpy as np
import pandas as pd
import joblib
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ----------------------------------------
# Create model directory if not exists
# ----------------------------------------
os.makedirs("model", exist_ok=True)


# ----------------------------------------
# Load Dataset
# ----------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset Shape:", X.shape)


# ----------------------------------------
# Train-Test Split
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ----------------------------------------
# Feature Scaling
# ----------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ----------------------------------------
# Evaluation Function
# ----------------------------------------
def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"\n{name}")
    print("Accuracy:", accuracy)
    print("AUC:", auc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("MCC:", mcc)


# ----------------------------------------
# Define Models
# ----------------------------------------
models = {
    "logistic_reg": LogisticRegression(max_iter=10000),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}


# ----------------------------------------
# Train, Evaluate and Save
# ----------------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    evaluate_model(model, name)
    joblib.dump(model, f"model/{name}.pkl")

# Save scaler
joblib.dump(scaler, "model/scaler.pkl")

print("\nAll models trained and saved successfully!")
