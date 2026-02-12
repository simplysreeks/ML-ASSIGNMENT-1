# Machine Learning Classification - Assignment 2

## Problem Statement

The objective of this project is to implement and compare multiple machine learning classification models on a real-world dataset. The goal is to evaluate model performance using various classification metrics and deploy the solution using Streamlit.

---

## Dataset Description

This project uses the **Breast Cancer Wisconsin Diagnostic Dataset**.

- Total Instances: 569
- Total Features: 30
- Target Variable: Binary Classification
    - 0 = Malignant
    - 1 = Benign

The dataset contains numerical features computed from digitized images of breast mass.

---

## Machine Learning Models Used

The following 6 classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

---

## Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-----------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.9737 | 0.9974 | 0.9722 | 0.9859 | 0.9790 | 0.9439 |
| Decision Tree | 0.9298 | 0.9299 | 0.9565 | 0.9296 | 0.9429 | 0.8526 |
| KNN | 0.9474 | 0.9820 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| Naive Bayes | 0.9649 | 0.9974 | 0.9589 | 0.9859 | 0.9722 | 0.9253 |
| Random Forest | 0.9649 | 0.9949 | 0.9589 | 0.9859 | 0.9722 | 0.9253 |
| XGBoost | 0.9561 | 0.9908 | 0.9583 | 0.9718 | 0.9650 | 0.9064 |

---

## Observations

| ML Model | Observation |
|------------|-------------|
| Logistic Regression | Achieved the highest overall performance with excellent AUC and MCC, showing strong linear separability in the dataset. |
| Decision Tree | Slightly lower performance due to possible overfitting on training data. |
| KNN | Performed well but sensitive to scaling and distance calculations. |
| Naive Bayes | High AUC score but assumes feature independence, which may not fully hold. |
| Random Forest | Stable and robust performance due to ensemble averaging. |
| XGBoost | Strong performance with good generalization, though slightly lower than Logistic Regression. |

---

## Streamlit Application Features

- CSV file upload option
- Model selection dropdown
- Display of all evaluation metrics
- Confusion matrix visualization

---

## Deployment

The application is deployed using Streamlit Community Cloud.

