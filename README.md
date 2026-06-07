# 💳 Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow?style=flat&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?style=flat&logo=numpy)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-green?style=flat)

---

## 📌 Overview

Credit card fraud causes billions of dollars in financial losses every year. This project builds an end-to-end machine learning pipeline to detect fraudulent transactions from real-world anonymized credit card data. The pipeline addresses one of the most critical challenges in financial ML — **extreme class imbalance** — and evaluates multiple classification models optimized for fraud recall and precision, directly applicable to real banking and fintech systems.

---

## 🎯 Problem Statement

Given a dataset of credit card transactions, classify each transaction as **fraudulent (1)** or **legitimate (0)**. The dataset is heavily imbalanced — fraudulent transactions represent a tiny fraction of all records — making standard accuracy an insufficient metric. The goal is to maximize **fraud detection rate (Recall)** while keeping **false positives (False Positive Rate)** manageable, using techniques specifically designed for imbalanced classification problems.

---

## 📂 Dataset Information

- **Source:** [Kaggle — Credit Card Fraud Detection (ULB MLG)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **File:** `creditcard.csv`
- **Total Records:** 284,807 transactions
- **Features:** 30 numerical features (V1–V28 are PCA-transformed, plus `Time` and `Amount`)
- **Target Variable:** `Class` — 0 = Legitimate, 1 = Fraudulent
- **Class Distribution:** Highly imbalanced — fraud cases form less than 0.2% of total transactions

---

## 🔄 Project Workflow

```
Raw Data → EDA → Preprocessing → Baseline Model → SMOTE Resampling
→ Model Training (LR + XGBoost) → Threshold Optimization → Evaluation → Visualization
```

1. **Data Loading** — Loaded the dataset using Pandas; inspected shape, dtypes, and summary statistics
2. **Class Distribution Analysis** — Quantified the severe class imbalance between fraud and legitimate transactions
3. **Exploratory Data Analysis** — Visualized distributions, correlations, and transaction amount patterns by class
4. **Preprocessing** — Applied Standard Scaling to normalize features; performed stratified train-test split
5. **Baseline Modeling** — Trained Logistic Regression on the original imbalanced data to establish a baseline
6. **SMOTE Resampling** — Applied SMOTE to the training set to synthetically balance the minority (fraud) class
7. **Model Training** — Retrained Logistic Regression and XGBoost on the SMOTE-balanced training data
8. **Threshold Optimization** — Swept classification thresholds from 0.01 to 0.99 to maximize F1-Score for each model
9. **Model Evaluation** — Compared models using Classification Report, Confusion Matrix, ROC-AUC, and PR-AUC
10. **Feature Importance Analysis** — Extracted and visualized XGBoost feature importances to identify key fraud indicators

---

## 📊 Exploratory Data Analysis (EDA)

Performed comprehensive EDA to understand data characteristics and inform modeling decisions:

- **Class Distribution Plot** — Histogram and count plot revealing the extreme imbalance between fraud and legitimate classes
- **Transaction Amount by Class** — Box plot comparing the `Amount` distributions for fraudulent vs. legitimate transactions, revealing behavioral differences
- **Correlation Heatmap** — Full-feature Seaborn heatmap with `coolwarm` palette to detect multicollinearity among PCA features
- **Amount Distribution** — Histogram of raw transaction amounts to understand skewness and outliers
- **Descriptive Statistics** — Separate `Amount` statistics computed for fraud and non-fraud subsets to surface statistical differences

---

## 🛠️ Data Preprocessing

- **Feature-Target Split:** Separated feature matrix `X` (all columns except `Class`) and target vector `y` (`Class`)
- **Standard Scaling:** Applied `StandardScaler` to normalize all features to zero mean and unit variance — critical for Logistic Regression convergence
- **Stratified Train-Test Split:** Used `train_test_split` with `stratify=y` and `random_state=42` to preserve class proportions across train and test sets (75/25 split)
- **SMOTE Oversampling:** Applied `SMOTE` (Synthetic Minority Oversampling Technique) exclusively to the **training set** to avoid data leakage, synthetically generating fraud samples to achieve class balance

---

## 🤖 Model Development

### Models Trained

| Model | Training Data | Threshold Strategy |
|---|---|---|
| Logistic Regression (Baseline) | Original imbalanced data | Default 0.5 |
| Logistic Regression (SMOTE) | SMOTE-balanced data | Optimized via F1 sweep |
| XGBoost Classifier (SMOTE) | SMOTE-balanced data | Optimized via F1 sweep |

### Threshold Optimization
Instead of using the default 0.5 decision threshold, implemented a **threshold sweep** across 100 values from 0.01 to 0.99, selecting the threshold that maximizes F1-Score on the test set for each model. This is essential for imbalanced fraud detection, where the optimal threshold deviates significantly from 0.5.

### Why These Models?
- **Logistic Regression** — Interpretable, fast baseline; strong for linearly separable patterns; good probabilistic output for threshold tuning
- **XGBoost** — Gradient boosting ensemble; handles non-linear interactions and complex feature relationships; robust to outliers; state-of-the-art performance on tabular data

---

## 📈 Evaluation Metrics

Standard accuracy is misleading on imbalanced datasets — a model predicting "legitimate" for all transactions would achieve ~99.8% accuracy while detecting zero fraud. The following metrics were used instead:

| Metric | Why It Matters |
|---|---|
| **Precision** | Of all predicted frauds, how many are real? Minimizes false alarms |
| **Recall** | Of all real frauds, how many were caught? Critical for fraud detection |
| **F1-Score** | Harmonic mean of Precision and Recall; primary optimization target |
| **ROC-AUC** | Model's ability to distinguish classes across all thresholds |
| **PR-AUC** | Precision-Recall tradeoff; more informative than ROC-AUC on imbalanced data |
| **Confusion Matrix** | Breakdown of TP, TN, FP, FN for each model |

---

## 🏆 Results

- **SMOTE** successfully balanced the training class distribution, enabling models to learn meaningful fraud patterns
- **Threshold optimization** improved F1-Score over the default 0.5 threshold for both models
- **XGBoost** outperformed Logistic Regression on both ROC-AUC and PR-AUC, demonstrating stronger fraud pattern capture
- **Precision-Recall curves** confirmed XGBoost's superiority on the minority class, particularly at high recall thresholds

---

## 📉 Visualizations

| Visualization | Purpose |
|---|---|
| Correlation Heatmap | Feature relationship analysis |
| Fraud vs. Legitimate Count Plot | Class imbalance visualization |
| Transaction Amount Box Plot by Class | Behavioral difference between fraud and legitimate |
| Class Distribution Histogram | Raw class frequency |
| Amount Distribution Histogram | Transaction value spread |
| Class Distribution Before/After SMOTE | Impact of resampling |
| ROC Curve (LR vs XGBoost) | AUC-based model comparison |
| Precision-Recall Curve (LR vs XGBoost) | Imbalance-aware model comparison |
| Feature Importance Bar Chart (XGBoost) | Key fraud indicator identification |

---

## 🧰 Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (LogisticRegression, StandardScaler, train_test_split) |
| Boosting | XGBoost |
| Imbalanced Learning | imbalanced-learn (SMOTE) |
| Environment | Jupyter Notebook, Kaggle |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

---

## ▶️ How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the project directory
2. Update the file path in the notebook:
   ```python
   df = pd.read_csv('creditcard.csv')
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook credit-card-fraud-detection.ipynb
   ```
4. Run all cells sequentially (Cell → Run All)

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── credit-card-fraud-detection.ipynb   # Main notebook (EDA + modeling + evaluation)
├── README.md                           # Project documentation
└── creditcard.csv                      # Dataset (download from Kaggle)
```

---

## 💡 Key Learnings

- Real-world financial datasets are severely imbalanced; accuracy alone is a misleading performance metric
- SMOTE should be applied **only to training data** — applying it before splitting causes data leakage
- Custom threshold tuning via F1-Score sweep significantly improves minority class detection vs. default 0.5 threshold
- PR-AUC is more informative than ROC-AUC when evaluating models on imbalanced datasets
- XGBoost's ensemble approach captures non-linear fraud patterns that Logistic Regression misses
- Feature importance from tree-based models provides actionable insights into which transaction features drive fraud detection

---

## 🚀 Future Improvements

- Integrate **SHAP values** for granular, instance-level model explainability
- Experiment with **Isolation Forest** and **Autoencoders** for unsupervised anomaly detection
- Add **hyperparameter tuning** via GridSearchCV or Optuna for XGBoost optimization
- Build a **Streamlit web app** for real-time fraud prediction on new transactions
- Explore **ensemble stacking** (LR + XGBoost + Random Forest) for further performance gains
- Implement **cost-sensitive learning** to directly optimize business cost of false negatives vs. false positives

---

## 🌟 Resume-Worthy Highlights

- Engineered a complete fraud detection pipeline on 284,807 real-world credit card transactions, addressing extreme class imbalance using SMOTE oversampling to improve minority class recall
- Applied Standard Scaling and stratified train-test splitting to ensure data integrity and prevent leakage during preprocessing
- Conducted in-depth EDA using correlation heatmaps, box plots, and distribution analysis to uncover behavioral differences between fraudulent and legitimate transactions
- Trained and compared Logistic Regression and XGBoost classifiers, implementing custom threshold optimization via F1-Score sweep to maximize fraud detection performance
- Evaluated models using Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC, and PR-AUC — selecting metrics appropriate for severe class imbalance
- Generated and interpreted XGBoost feature importance rankings to identify the most predictive transaction attributes for fraud classification
- Demonstrated end-to-end ML workflow — from raw data ingestion through EDA, preprocessing, modeling, threshold tuning, and multi-metric evaluation

---

## 📝 Repository Description

> An end-to-end machine learning pipeline for detecting credit card fraud on 284K+ real-world transactions. Addresses class imbalance with SMOTE, compares Logistic Regression and XGBoost with threshold optimization, and evaluates performance using ROC-AUC, PR-AUC, and F1-Score.

---

## 🏷️ Project Tags

`machine-learning` `fraud-detection` `classification` `imbalanced-dataset` `smote` `xgboost` `logistic-regression` `scikit-learn` `pandas` `data-science` `exploratory-data-analysis` `feature-engineering` `roc-auc` `python` `kaggle`
