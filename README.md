
# Credit Card Fraud Detection

Binary classification on a highly imbalanced dataset: only **492 frauds out of 284,807 transactions (0.17%)**.

Dataset: [Credit Card Fraud Detection (Kaggle, mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Features V1-V28 are PCA components (anonymized), plus `Time`, `Amount` and the target `Class`.

## Why accuracy is misleading
A baseline Logistic Regression scores ~99.9% accuracy, yet it misses ~36% of the frauds (recall 0.64). With 99.8% of rows being normal, accuracy hides the failures. So models are compared on **recall, precision and F1 for the fraud class**.

## Approach
1. Data checks (missing values, duplicates, class imbalance)
2. Stratified 80/20 train/test split (test set: 98 frauds)
3. `RobustScaler` on `Amount` and `Time`, fit on train only (no leakage)
4. Models: Logistic Regression (baseline), Logistic Regression with `class_weight="balanced"`, Random Forest, XGBoost

## Results (fraud class, test set)

| Model | Recall | Precision | F1 |
|---|---|---|---|
| LR baseline | 0.64 | 0.83 | 0.72 |
| LR balanced | 0.92 | 0.06 | 0.11 |
| Random Forest | 0.82 | 0.94 | 0.87 |
| XGBoost | 0.80 | 0.91 | 0.85 |

## Key takeaways
- `class_weight="balanced"` raised recall (0.64 to 0.92) but flooded the model with false alarms (precision 0.06). It only moves the recall/precision trade-off, it does not make the model better.
- Tree models improved recall and precision together (F1 0.72 to 0.85-0.87).
- Random Forest and XGBoost are comparable: with only 98 test frauds, a 0.02 F1 gap is about 2 transactions.

## Limitations / future work
- SMOTE, threshold tuning and PR-AUC analysis
- Cross-validation for more reliable estimates
- Serving the model with FastAPI (`/predict`)

## Stack
Python, pandas, scikit-learn, XGBoost, joblib
