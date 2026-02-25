# Credit Card Fraud Detection: End-to-End MLOps on Azure

[![Azure ML](https://img.shields.io/badge/Azure-Machine%20Learning-blue)](https://azure.microsoft.com/en-us/services/machine-learning/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-orange)](https://mlflow.org/)

## 📌 Project Overview
In financial services, the cost of a "Missed Fraud" (False Negative) is significantly higher than a "False Alarm" (False Positive). This project implements a production-ready fraud detection pipeline on **Azure Machine Learning**, focusing on optimizing **Recall** to protect revenue.

### 🎯 Key Performance Results
| Metric | Value | Business Significance |
| :--- | :--- | :--- |
| **Recall** | **82.6%** | Successfully identified the vast majority of fraudulent transactions. |
| **Precision** | **88.0%** | Maintained high trust by minimizing unnecessary transaction blocks. |
| **AUPRC** | **0.88** | Demonstrated robust handling of extreme class imbalance (0.17% fraud). |

---

## 🛠️ Technical Stack & Architecture
* **Cloud Platform:** Azure Machine Learning (Compute Instances & Clusters)
* **Model Management:** **MLflow** for experiment tracking and model registration.
* **Algorithms:** XGBoost and Random Forest with **SMOTE** (Synthetic Minority Over-sampling Technique) to address class imbalance.
* **Data Governance:** Standardized version control and .amlignore configurations for clean cloud synchronization.

---

## 📂 Repository Structure
* **01_set_up_workspace.ipynb**: Configuration of Azure ML Workspace, Datastore, and Compute resources.
* **02_eda_and_preprocessing.ipynb**: Statistical analysis of transaction patterns and feature engineering.
* **03_training_and_evaluation.ipynb**: Hyperparameter tuning, model comparison, and logging metrics to MLflow.
* **credit_card_fraud_detection.ipynb**: The consolidated "Master Pipeline" for the entire workflow.
* **requirements.txt**: Defined environment dependencies for reproducible deployment.

---

## 🚀 Key MLOps Features
* **Reproducible Environments:** Uses a curated requirements.txt to ensure consistent model behavior.
* **Experiment Tracking:** Every run logs parameters and loss curves via MLflow.
* **Balanced Sampling:** Implemented SMOTE within the training pipeline.

---

## 🔧 Installation & Usage
1. **Clone the repo:** git clone git@github.com:cyfangus/credit_card_fraud_detection_AzureML.git
2. **Install dependencies:** pip install -r requirements.txt
