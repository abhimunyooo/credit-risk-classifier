# Credit Risk Classification on Imbalanced Data

## Overview

This project demonstrates how to handle **severely imbalanced classification problems** using a real-world credit card transactions dataset.  
The primary objective is to **understand and compare different class imbalance handling techniques** and observe their impact on model performance, especially on the minority (high-risk) class.

Rather than focusing on model complexity, this project emphasizes:
- Correct evaluation metrics
- Data leakage prevention
- Business-oriented interpretation of results

---

## Problem Statement

In credit risk and fraud detection problems, the dataset is often **extremely imbalanced**, where risky or fraudulent cases form a very small fraction of the data.

In this dataset:
- Class `0` → Normal / Low-risk transactions  
- Class `1` → Fraudulent / High-risk transactions  
- Minority class proportion ≈ **0.17%**

A naive model can achieve very high accuracy by predicting only the majority class, while completely failing at detecting risky cases.  
This project explores why that happens and how to mitigate it.

---

## Dataset

- Source: Credit Card Transactions Dataset (publicly available)
- Size: **284,807 rows × 31 columns**
- Features:
  - `V1`–`V28`: PCA-transformed features
  - `Time`: Time elapsed since first transaction
  - `Amount`: Transaction amount
  - `Class`: Target variable (0 = normal, 1 = high-risk)

No missing values are present in the dataset.

---

## Project Workflow

The notebook follows a **strict, step-by-step pipeline**:

1. **Data loading and validation**
2. **Class imbalance analysis**
3. **Minimal exploratory data analysis**
4. **Stratified train–test split**
5. **Robust scaling (Time & Amount only)**
6. **Baseline Logistic Regression (no resampling)**
7. **Random undersampling**
8. **SMOTE oversampling**
9. **NearMiss undersampling**
10. **Side-by-side comparison and conclusions**

Each resampling technique is applied **only on the training data**, and all models are evaluated on the **same untouched test set**.

---

## Models Used

- **Logistic Regression**
  - Chosen intentionally for interpretability
  - Makes the effect of class imbalance clearly visible
  - No class weights or hyperparameter tuning applied

---

## Evaluation Metrics

Because accuracy is misleading for imbalanced datasets, the following metrics are emphasized:

- Recall (Class 1 / High-risk)
- Precision (Class 1 / High-risk)
- Confusion Matrix
- ROC-AUC (for probability ranking insight)

---

## Key Results (Summary)

| Method | Recall (High-risk) | Precision (High-risk) | False Positives |
|------|-------------------|-----------------------|----------------|
| No Resampling | ~64% | ~83% | Very Low |
| Random Undersampling | ~92% | ~3.8% | High |
| **SMOTE** | **~92%** | **~5.9%** | Moderate |
| NearMiss | ~96% | ~0.4% | Extremely High |

---

## Conclusions

- **Accuracy alone is not a valid metric** for imbalanced classification problems.
- Random undersampling and NearMiss significantly increase recall but at the cost of excessive false positives.
- **SMOTE provides the best balance** between detecting high-risk cases and preserving majority-class information.
- Logistic Regression is sufficient to demonstrate imbalance effects clearly; more complex models are not required for this analysis.

---

## What This Project Does NOT Cover

The following were intentionally excluded to keep the focus clear:
- Anomaly detection methods (Isolation Forest, LOF)
- Neural networks
- Cross-validation
- Dimensionality reduction for modeling (PCA / t-SNE)
- Hyperparameter tuning

These can be added as extensions but are not required to understand class imbalance handling.

---

## How to Run

1. Clone the repository
2. Install dependencies

```bash
   pip install -r requirements.txt
```

3. Open the notebook and run cells sequentially

---

## Intended Audience

* Machine learning learners
* Data science students
* Engineers preparing for interviews involving imbalanced datasets
* Anyone looking to understand **why resampling matters more than model complexity**

---

## License

This project is for educational purposes.