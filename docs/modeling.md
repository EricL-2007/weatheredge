# Modeling

## Classification
Target:
- rain_tomorrow

Models:
- Baseline dummy
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

Metrics:
- Accuracy
- Precision
- Recall
- F1
- ROC AUC
- Average precision

## Regression
Target:
- next-period temperature

Models:
- Linear Regression
- Random Forest
- XGBoost
- LightGBM

Metrics:
- MAE
- RMSE
- R^2

## Calibration
Methods:
- Platt scaling
- Isotonic regression

Metrics:
- Brier score
- ECE
- Log loss

## Explainability
- Global feature importance
- Local SHAP values
