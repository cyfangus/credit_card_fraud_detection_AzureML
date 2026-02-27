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
<img width="567" height="202" alt="Screenshot 2026-02-27 at 12 59 07" src="https://github.com/user-attachments/assets/1d7f18e2-d329-4c6c-8f5d-35ad8a36febc" />

### ⚖️ Strategic Model Selection: The "Risk vs. Friction" Narrative
My selection of XGBoost-SMOTE as the champion model represents a strategic balance between security coverage and operational viability. While Logistic Regression achieved the highest nominal **Recall (0.918)**, its critically **low Precision (0.054)** would result in an unsustainable 18:1 false-positive ratio, causing massive customer friction and overwhelming manual review teams. In contrast, XGBoost delivered the optimal "Golden Ratio": capturing **87.7% of fraud (Recall)** while providing a **7x improvement in Precision** over Logistic Regression. Furthermore, the technical efficiency of the XGBoost architecture on Azure was decisive; it completed the SMOTE-enhanced training in just **8 seconds, a 21x speed** advantage over the Random Forest equivalent (168s). This agility ensures that the model can be retrained and redeployed in minutes to counteract evolving adversarial tactics in real-time.

## 🔬 Model Optimization & Refinement
To maximize the model's discriminative power, I conducted Hyperparameter Optimization using RandomizedSearchCV with 3-fold Cross-Validation.

### The Strategy:
- Target Metric: Optimized for AUPRC (Average Precision) rather than accuracy to prioritize the Precision-Recall trade-off essential for fraud detection.
- Regularization: Focused on reg_lambda (L2) and reg_alpha (L1) to prevent the model from overfitting to synthetic SMOTE artifacts.
- Efficiency: Utilized a 10-iteration search to identify a high-performing configuration while maintaining computational efficiency.

Winning Parameters:
| Parameter | Value | Impact |
| :--- | :--- | :--- |
| max_depth | 6 | Balanced model complexity with generalization. |
| reg_lambda | 5 | Applied strong L2 penalty to stabilize weights. |
| learning_rate | 0.1 | Ensured robust convergence during boosting. |

## Precision-Recall Evolution: A Comparative Analysis
<img width="637" height="624" alt="Screenshot 2026-02-27 at 13 47 38" src="https://github.com/user-attachments/assets/22db9bcb-0d48-4bf8-8c0e-bbe09407bf33" />
Before finalizing the operational policy, I analyzed the Precision-Recall (PR) Curve to quantify the impact of oversampling and hyperparameter tuning. Unlike ROC curves, which can be overly optimistic on imbalanced data, the PR curve provides a transparent view of the model's ability to identify rare fraud events while minimizing customer friction. While the Average Precision (AP) slightly adjusted after tuning (dark blue), the curve became more stable. This indicates better generalization and a more reliable "elbow" for selecting operational thresholds.

## Threshold tuning
<img width="402" height="200" alt="Screenshot 2026-02-27 at 14 00 27" src="https://github.com/user-attachments/assets/a9d6f7e3-e117-4cf9-8a3e-b9693a8d3d7c" />
Following Hyperparameter Optimization, I conducted a Strategic Threshold Analysis to define the model's operational policy. The tuned XGBoost model demonstrated exceptional calibration, allowing for a Balanced Policy (0.950 threshold) that captures 81.6% of fraud with an 88.9% Precision rate. This configuration effectively minimizes operational overhead by limiting false positives to just 10 cases, representing a highly efficient 'Precision-First' deployment strategy for real-time transaction monitoring.

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
