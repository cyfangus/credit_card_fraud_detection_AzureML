# Fraud Detection Strategy: Adversarial Pattern Analysis on Azure

[![Azure ML](https://img.shields.io/badge/Azure-Machine%20Learning-blue)](https://azure.microsoft.com/en-us/services/machine-learning/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-orange)](https://mlflow.org/)

## 📌 Executive Summary
In the financial technology sector, the primary challenge of fraud detection lies in the extreme imbalance of data and the adversarial nature of criminal behavior. This project demonstrates a production-ready solution for identifying fraudulent transactions within a dataset where the incident rate is only **0.17%**.

Using an end-to-end pipeline on **Azure Machine Learning Studio**, I developed a model that prioritizes **Recall (82.6%)** over simple accuracy. This ensures maximum disruption of illicit activity while maintaining the precision necessary to preserve user trust and minimize friction for legitimate customers.

---

## ⚖️ The Operational Problem: Imbalanced Data
Standard machine learning models often fail in fraud contexts because they optimize for overall accuracy, which can lead to a 100% failure rate in detecting the minority "fraud" class. 

* **Dataset:** 284,807 transactions.
* **Incident Rate:** 492 fraudulent cases (0.17%).
* **Objective:** Developing a robust detection mechanism that identifies high-risk signatures without disrupting the "frictionless" travel and payment experience required by high-growth platforms.

<img width="867" height="570" alt="image" src="https://github.com/user-attachments/assets/1923e780-edc8-4ee2-a3e4-7066314a0388" />


---

## 🛠️ Technical Methodology

### Data Engineering and Imbalance Mitigation
To address the severe class imbalance, I implemented **SMOTE (Synthetic Minority Over-sampling Technique)**. Unlike simple duplication, SMOTE creates synthetic examples based on the feature space of existing fraud cases. This allows the classifier to learn the structural "shape" of adversarial behavior rather than just memorizing specific instances.


### Feature Selection and Behavioral Analysis
Leveraging my background in **Crime Science**, I performed a deep-dive analysis into the Principal Component (PC) features. I identified specific behavioral clusters—often related to transaction velocity and geographical anomalies—that served as the primary indicators of fraudulent intent.

### Professional MLOps Workflow
The solution was engineered for scalability and production-readiness:
* **Environment:** Azure Machine Learning Studio.
* **Optimization:** Hyperparameter tuning using automated sweeps to find the optimal balance between precision and recall.
* **Deployment:** Registered the model and deployed it as a **REST-based real-time scoring endpoint**, mimicking the latency requirements of a live payment gateway.

---

## 📈 Performance and Business Impact

| Metric | Performance | Operational Value |
| :--- | :--- | :--- |
| **Recall** | **82.6%** | High sensitivity for capturing 4 out of 5 fraud attempts. |
| **AUPRC** | **0.88** | Demonstrated stability across varying risk thresholds. |
| **Status** | **Deployed** | API-ready for integration into real-time risk engines. |



### Strategic Implementation: Recall vs. Precision
As a security researcher, I advocate for a **Risk-Proportional Response**:
1. **High-Confidence Fraud:** Automatic block and referral to manual investigative teams.
2. **Medium-Risk Anomalies:** Stepped verification (3D Secure or SMS MFA) to maintain user trust while mitigating potential losses.

---

## 📂 Repository Structure
* `/data`: Dataset documentation and schema definitions.
* `/notebooks`: Exploratory Data Analysis (EDA) and feature importance visualizations.
* `/src`: Production-ready Python scripts for training and scoring.
* `/deployment`: Configuration files for Azure ML Service endpoints.

---

## 🤝 Contact and Portfolio
**Angus Chan** PhD Researcher | Security and Crime Science | UCL  
[LinkedIn Profile] | [Technical Portfolio]
