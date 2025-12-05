# Training and Evaluation

This document explains how to run model training and where to find evaluation metrics.

Files
- `fastapi_ml_service/train_models.py`: training script for distance, human detection, and location classification.

Output
- Models saved to `fastapi_ml_service/models/` as `.pkl` files.
- Metrics saved to `fastapi_ml_service/models/*_metrics.json`.

Run training

```powershell
# Activate your venv first
python .\fastapi_ml_service\train_models.py
```

What the script does
- Loads cleaned CSVs from `cleaned_data/` (folders `radio_combined` and `radio_separate`).
- Extracts features and derived features (normalization, squared terms, composite `signal_quality`).
- Trains RandomForest models with fixed hyperparameters (see script for details).
- Evaluates on test set and prints metrics to console.
- Runs 5-fold cross-validation and saves mean/std in the metrics JSON.

Interpreting metrics
- Regression (distance): `mae`, `rmse`, `r2`, and `cv_mae_mean`/`cv_mae_std`.
- Classification: `accuracy`, `cv_accuracy_mean`, `cv_accuracy_std`, and `classification_report` printed to console.

Improving performance
- More feature engineering (temporal features, variance, deltas).
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV).
- Try other models (XGBoost, LightGBM, SVM) and ensemble approaches.
