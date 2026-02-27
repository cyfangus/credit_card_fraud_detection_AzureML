# Fraud Detection Strategy: Adversarial Pattern Analysis on Azure

[![Azure ML](https://img.shields.io/badge/Azure-Machine%20Learning-blue)](https://azure.microsoft.com/en-us/services/machine-learning/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-orange)](https://mlflow.org/)


## 📌 Executive Summary
In the financial technology sector, the primary challenge of fraud detection lies in the extreme imbalance of data and the adversarial nature of criminal behavior. This project demonstrates a production-ready solution for identifying fraudulent transactions within a dataset where the incident rate is only **0.17%**.

Using an end-to-end pipeline on **Azure Machine Learning Studio**, I developed a multi-layered defense system that prioritizes **Recall (86.7%)** to ensure maximum disruption of illicit activity while providing the interpretability required for professional investigative workflows.

---

## ⚖️ The Operational Problem: Imbalanced Data
Standard machine learning models often fail in fraud contexts because they optimize for overall accuracy, which can lead to a 100% failure rate in detecting the minority "fraud" class. 

* **Dataset:** 284,807 transactions.
* **Incident Rate:** 492 fraudulent cases (0.17%).
* **Objective:** Developing a robust detection mechanism that identifies high-risk signatures without disrupting the frictionless payment experience required by high-growth platforms like **Duffel** or **Spotify**.


<img width="867" height="570" alt="image" src="https://github.com/user-attachments/assets/1923e780-edc8-4ee2-a3e4-7066314a0388" />

---

## 🛠️ Technical Methodology: The Hybrid Defense Strategy

To provide comprehensive coverage, I implemented a two-layer approach that addresses both known behavioral signatures and novel, "zero-day" attack vectors.

### Layer 1: Supervised Learning (XGBoost + SMOTE)
To handle the severe class imbalance, I implemented **SMOTE (Synthetic Minority Over-sampling Technique)** to synthetically generate fraud examples for training. While Random Forest provided a strong baseline, I selected **XGBoost** for production due to its superior performance in a security context:
* **High-Recall Focus:** Captured **86.7%** of fraud cases (a 4.1% improvement over Random Forest).
* **Operational Speed:** Achieved a **30x reduction** in training time on Azure (9s vs 269s), enabling rapid model iteration as fraud tactics evolve.

### Layer 2: Unsupervised Anomaly Detection (Isolation Forest)
To defend against novel attacks that lack historical labels, I integrated an **Isolation Forest** layer. This identifies outliers based on structural anomalies in the transaction distribution, acting as a "safety net" for emerging fraud patterns that supervised models might miss.

---

## 📈 Performance and Model Selection

| Algorithm | Strategy | Recall | Duration | AUPRC |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **SMOTE** | **0.867** | **9s** | **0.871** |
| Random Forest | SMOTE | 0.826 | 269s | 0.881 |
| Logistic Regression | SMOTE | 0.918 | 14s | 0.727 |

### ⚖️ Decision Logic: The Business vs. Technical Trade-off
Selecting the production model required a multi-dimensional evaluation beyond simple accuracy. My decision to **move forward with XGBoost-SMOTE** was based on two primary pillars:

1. The Risk-Adjusted Cost of Error (Business Case)
- In Fraud Analytics, the "Cost of a Missed Fraud" (False Negative) is significantly higher than the "Cost of a False Alarm" (False Positive).

- The Recall Priority: Missing a fraudulent transaction results in direct financial loss and potential regulatory scrutiny. XGBoost achieved **the highest Recall (0.867)**, successfully identifying **4.1% more fraud than Random Forest**.

- Operational Handling: While Random Forest had higher precision, the lower precision of XGBoost is operationally manageable. In a fintech workflow, these "extra" flags would be routed to a low-friction Step-up Authentication (e.g., SMS/Push notification), protecting the bottom line without permanently blocking legitimate users.

2. Computational Efficiency & Agility (Technical Case)
- Fraud is an adversarial race. The ability to retrain and deploy models faster than a fraudster can pivot is a massive competitive advantage.

- Training Latency: On Azure ML, XGBoost-SMOTE completed the training cycle in 9 seconds, compared to 269 seconds for Random Forest.

- Agility: This **30x speed improvement** allows for rapid hyperparameter tuning and near-instantaneous model updates. In a production environment like Duffel, this efficiency reduces cloud compute costs and enables a "CI/CD for ML" (MLOps) approach that keeps the defense system ahead of emerging threats.

---

## 🔍 Model Debugging and Interpretability

* **Global Importance:** Identified that features **V17** and **V14** are the primary drivers of risk detection, which I interpret as high-velocity behavioral anomalies.
* **Local Interpretability (SHAP):** The system provides "reason codes" for every flagged transaction. Using SHAP waterfall plots, investigators can see exactly which feature combination (e.g., amount + location) triggered a specific high-risk score.
* **Error Analysis:** Performed **Error Tree Analysis** on Azure to identify "blind spots," discovering that model performance diverges in low-value transaction cohorts, allowing for targeted rule-based mitigations.

---

## 📂 Repository Structure
* `/data`: Dataset documentation and schema definitions.
* `/notebooks`: Exploratory Data Analysis (EDA) and SHAP interpretability visualizations.
* `/src`: Production-ready Python scripts for training (XGBoost/SMOTE).
* `/deployment`: Azure ML configuration files for REST-based real-time scoring.

---

## 🤝 Contact and Portfolio
**Angus Chan** PhD Researcher | Security and Crime Science | UCL  
[LinkedIn Profile] | [Technical Portfolio]
