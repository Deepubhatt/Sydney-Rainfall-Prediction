# Sydney Rainfall Prediction

Predicting next-day rainfall in Sydney using classification and ensemble learning models, with model comparison and hyperparameter tuning.

## Problem

Given ten years (2008–2017) of daily weather observations for Sydney, predict whether it will rain the next day (`RainTomorrow`). This is a binary classification problem on tabular weather data with the usual real-world mess: missing values, skewed features, and correlated predictors.

## Dataset

- **Source:** Sydney weather station records, 2008–2017 (Kaggle "Rain in Australia" dataset, filtered to Sydney)
- **Size:** ~3,300 rows, 18 columns
- **Target:** `RainTomorrow` (1 = rain next day, 0 = no rain)
- **Features:** temperature, rainfall, evaporation, sunshine, humidity, pressure, cloud cover, and same-day rain indicator, recorded at 9am and 3pm

Full column descriptions are in the notebook.

## Approach

1. **Cleaning:** impute numeric columns with mean, categorical columns with mode
2. **Feature engineering:** extract month from date, one-hot encode categorical fields, drop `Date` and `Location`
3. **Outlier treatment:** log-transform `Rainfall` to fix right-skew, IQR-based capping for other numeric columns
4. **Multicollinearity:** drop features highly correlated with other predictors, based on a correlation heatmap
5. **Modeling:** train and evaluate 9 classifiers, then tune the top candidates with `GridSearchCV`
6. **Evaluation:** compare accuracy, F1, and ROC-AUC; confusion matrix per model

## Models compared

Logistic Regression, LDA, KNN, Decision Tree, Bagging, Random Forest, Gradient Boosting, AdaBoost, XGBoost.

| Model | Accuracy (default) | Accuracy (tuned) |
|---|---|---|
| Random Forest | 84.73% | — |
| Gradient Boosting | 83.98% | 84.43% |
| XGBoost | 83.68% | **85.03%** |
| AdaBoost | — | 84.88% |
| KNN | 79.79% | 82.93% |

*(Full per-model table, confusion matrices, and ROC-AUC scores are in the notebook.)*

## Result

**Tuned XGBoost** performed best at **85.03% accuracy**. It benefits from sequential boosting that corrects prior errors, built-in L1/L2 regularization that controls overfitting, and native handling of the dataset's mixed feature scales. `Pressure9am` and `Temp3pm` came out as the most predictive features.

## Possible improvements

- Feature engineering: interaction terms between humidity, pressure, and temperature
- Handle class imbalance in `RainTomorrow` (SMOTE or class weighting) rather than relying on accuracy alone
- Try LightGBM / CatBoost for comparison
- Time-based train/test split instead of random split, since this is sequential weather data
- SHAP values for feature importance instead of default XGBoost importance

## Run it locally

```bash
git clone https://github.com/<your-username>/sydney-rainfall-prediction.git
cd sydney-rainfall-prediction
pip install -r requirements.txt
jupyter notebook notebooks/sydney_rain_prediction.ipynb
```

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, statsmodels, seaborn, matplotlib

## License

MIT
