# 💳 Building a 2-layered Credit Card Fraud Detection Frameowork on AzureML
**An MLOps Case Study in Extreme Class Imbalance (0.17%)**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Azure ML](https://img.shields.io/badge/Cloud-Azure%20ML%20SDKv2-orange.svg)](https://azure.microsoft.com/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-blueviolet.svg)](https://mlflow.org/)

---

### 🎯 Business Value Proposition
In modern payment processing, **False Positives** are as expensive as fraud itself. Blocking a legitimate customer during a purchase results in immediate revenue loss, support overhead, and long-term churn. 

This project delivers a **Frictionless Fraud Engine** that achieves a **77.5% Precision rate**—ensuring that when the system flags a transaction, it is highly likely to be fraudulent, thereby protecting the user experience while maintaining elite security standards.

---

## ⚖️ The Operational Problem: Imbalanced Data
Standard machine learning models often fail in fraud contexts because they optimize for overall accuracy, which can lead to a 100% failure rate in detecting the minority "fraud" class. 

* **Dataset:** 284,807 transactions.
* **Incident Rate:** 492 fraudulent cases (0.17%).
* **Objective:** Developing a robust detection mechanism that identifies high-risk signatures without disrupting the frictionless payment experience of customers.


<img width="867" height="570" alt="image" src="https://github.com/user-attachments/assets/1923e780-edc8-4ee2-a3e4-7066314a0388" />

---

## 🛠️ Technical Methodology: The Hybrid Defense Strategy
To provide comprehensive coverage, I implemented a dual-layered architectural approach. This hybrid strategy addresses both known behavioral signatures (historical patterns) and novel, "zero-day" attack vectors (unseen anomalies).

### Layer 1: Supervised Learning (The "Pattern Matcher")
The primary layer utilizes a supervised framework to map known fraud signatures. I conducted a systematic benchmark of Logistic Regression, Random Forest, and XGBoost to identify the most robust engine for this high-dimensional PCA feature space.

#### The Precision-Recall Optimization: In fraud detection, the goal is rarely "Accuracy." Instead, I focused on the Precision-Recall Trade-off. A model with high Recall but low Precision creates "Customer Friction" by blocking legitimate users. My methodology involved tuning the decision boundary to maximize AUPRC (Area Under Precision-Recall Curve), ensuring the system can catch fraud without compromising the user experience.

#### Operational Efficiency: Beyond predictive power, I evaluated each candidate on Inference Latency. In a production payment gateway, a model must return a verdict in milliseconds. I specifically tested how each algorithm handles Cost-Sensitive Learning (via scale_pos_weight and class_weight) to manage the 0.17% imbalance without the overhead of synthetic data generation.

### Layer 2: Unsupervised Anomaly Detection (The "Safety Net")
To defend against "Cold Start" fraud—novel attacks that lack historical labels—I integrated an Isolation Forest layer.

#### Structural Outliers: Unlike supervised models that look for "Fraud like the past," Isolation Forest identifies transactions that are simply "not like the rest" by isolating anomalies in high-dimensional space.

#### Zero-Day Protection: This layer acts as a heuristic safety net. If a transaction appears structurally anomalous but passes the supervised check, it can be flagged for "Manual Review" or "Step-up Authentication" (MFA) rather than being blindly approved.

---

### 🏆 Results: The Model Tournament
I bypassed traditional synthetic resampling (SMOTE) in favor of **Cost-Sensitive Algorithmic Weighting** to preserve the integrity of the original feature space and minimize synthetic noise.

| Candidate Algorithm | Strategy | AUPRC (Gold Standard) | Precision (User Friction) | Recall (Detection Rate) |
| :--- | :--- | :--- | :--- | :--- |
| **🏆 XGBoost** | **Cost-Sensitive** | **0.861** | **0.775** | **0.846** |
| Random Forest | Balanced Weights | 0.828 | 0.778 | 0.826 |
| Logistic Regression | Balanced Weights | 0.723 | 0.055 | 0.918 |



---

### 🧠 Strategic Technical Decisions

#### 1. Rejecting SMOTE for "Data Integrity"
While SMOTE is popular in academic settings, I rejected it for this production-simulated environment. Synthetic data in high-dimensional PCA space often "smears" decision boundaries, leading to an unacceptable drop in Precision. By using XGBoost’s `scale_pos_weight`, I optimized the model for the **actual** 0.17% distribution, yielding a significantly more reliable model for real-world deployment.

#### 2. Advanced Regularization & Hyperparameter Tuning
Using `RandomizedSearchCV` integrated with **MLflow Autologging**, I optimized the XGBoost champion for generalization across 15 iterations.
* **Regularization:** Tuned `reg_alpha` and `reg_lambda` to penalize complexity and prevent overfitting.
* **Robustness:** Optimized `subsample` and `max_depth` to ensure the model captures patterns rather than specific "outlier" transactions.

#### 3. Threshold Optimization: The "Business Dial"
I implemented a post-training **Threshold Tuning** phase. This moves beyond the default 0.5 probability to find the mathematical "sweet spot" for business operations.
* **Standard Threshold (0.5):** Often too aggressive for frictionless payments.
* **Optimized Threshold:** Adjusted to maximize the **F1-Score**, providing a data-driven balance between fraud capture and false alarms.



---

### 🛠️ MLOps & Infrastructure
* **Azure ML SDK v2:** Orchestrated the training lifecycle on cloud-scalable compute.
* **MLflow Tracking:** Captured every metric (AUPRC, Recall, F1) and parameter to ensure 100% reproducibility and experiment transparency.
* **Artifact Management:** Models were logged as standard serialized artifacts, ensuring stability across diverse cloud deployment environments.

---

### 📈 Visualizing Performance
*The Precision-Recall curve below illustrates the model's performance. The area under the curve (0.86) demonstrates an elite ability to rank fraud above legitimate transactions even in highly skewed datasets.*



---

### 💡 Key Takeaways for Recruiters
* **Imbalance Expertise:** Proven ability to handle extreme skewness (0.17%) using modern cost-sensitive techniques.
* **Metric Focused:** Priority placed on **AUPRC** and **Precision**, reflecting the actual financial success metrics of a Fintech organization.
* **Production Ready:** Clean, modular code tracked with enterprise-grade MLOps tools (Azure ML & MLflow).

---

### 🏁 How to Use
1. Clone the repository.
2. Ensure your Azure ML `config.json` is present in the root directory.
3. Execute `03_training_&_deployment.ipynb` to reproduce the tournament and tuning results.


# 💳 Credit Card Fraud Detection: Precision-Engineered Security on Azure ML


[![Azure ML](https://img.shields.io/badge/Azure-Machine%20Learning-blue)](https://azure.microsoft.com/en-us/services/machine-learning/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-orange)](https://mlflow.org/)


## 📌 Executive Summary
In the financial technology sector, the primary challenge of fraud detection lies in the extreme imbalance of data and the adversarial nature of criminal behavior. This project demonstrates a production-ready solution for identifying fraudulent transactions within a dataset where the incident rate is only **0.17%**.

Using an end-to-end pipeline on **Azure Machine Learning Studio**, I developed a multi-layered defense system that prioritizes **Recall (86.7%)** to ensure maximum disruption of illicit activity while providing the interpretability required for professional investigative workflows.




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
<<<<<<< HEAD
<img width="637" height="624" alt="Screenshot 2026-02-27 at 13 47 38" src="https://github.com/user-attachments/assets/22db9bcb-0d48-4bf8-8c0e-bbe09407bf33" />
=======
<img width="400" height="400" alt="Screenshot 2026-02-27 at 13 47 38" src="https://github.com/user-attachments/assets/22db9bcb-0d48-4bf8-8c0e-bbe09407bf33" />
>>>>>>> 10d3263 (added 2 layer model)

Before finalizing the operational policy, I analyzed the Precision-Recall (PR) Curve to quantify the impact of oversampling and hyperparameter tuning. Unlike ROC curves, which can be overly optimistic on imbalanced data, the PR curve provides a transparent view of the model's ability to identify rare fraud events while minimizing customer friction. While the Average Precision (AP) slightly adjusted after tuning (dark blue), the curve became more stable. This indicates better generalization and a more reliable "elbow" for selecting operational thresholds.

## Threshold tuning
<img width="402" height="200" alt="Screenshot 2026-02-27 at 14 00 27" src="https://github.com/user-attachments/assets/a9d6f7e3-e117-4cf9-8a3e-b9693a8d3d7c" />
Following Hyperparameter Optimization, I conducted a Strategic Threshold Analysis to define the model's operational policy. The tuned XGBoost model demonstrated exceptional calibration, allowing for a Balanced Policy (0.950 threshold) that captures 81.6% of fraud with an 88.9% Precision rate. This configuration effectively minimizes operational overhead by limiting false positives to just 10 cases, representing a highly efficient 'Precision-First' deployment strategy for real-time transaction monitoring.

## Adding Isolation Forest as 2nd layer
<img width="529" height="261" alt="Screenshot 2026-02-27 at 14 17 04" src="https://github.com/user-attachments/assets/7b8293c4-cfd2-4cd8-87ec-697af7ba994c" />

The implementation of the Isolation Forest as a secondary defense layer successfully identified 16.67% (3 out of 18) of the fraud cases missed by the supervised XGBoost model. By specifically targeting transactions with negative anomaly scores ($IF\_Score < 0$), the system identified "Silent Frauds" that had been classified as low-risk by the primary model. This unsupervised layer demonstrated a critical ability to catch novel, "Zero-Day" patterns that do not match historical profiles, most notably isolating a high-priority outlier (Index 9179) that the XGBoost model had assigned a fraud probability of only 6.4%.

---

## Conclusion: Multi-Layer Security vs. Business Friction
By implementing a dual-layer detection engine, this project achieved a high-precision state that balances security with customer experience. The Supervised Layer (XGBoost) acts as the primary shield, capturing the majority of known fraud patterns with 88.9% precision. The Unsupervised Layer (Isolation Forest) serves as a critical fail-safe, recovering 16.6% of frauds missed by the primary model—specifically those that exhibited high-entropy outlier behavior.

## Key Outcome: The final architecture allows the business to automate blocks on 80%+ of fraud while virtually eliminating false alarms (only 10 false positives in the test set), directly reducing operational costs associated with manual reviews and customer disputes.

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
