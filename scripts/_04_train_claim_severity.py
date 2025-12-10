# scripts/04_train_claim_severity.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from scripts.severity_pipeline import build_preprocessor, save_preprocessor

OUT_DIR = Path("../models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = Path("../reports/task4")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train_and_evaluate(train_path="../data/processed/severity_train.parquet",
                       test_path="../data/processed/severity_test.parquet",
                       target_col="TotalClaims",
                       use_log=True,
                       random_state=42):
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    # Build preprocessor and fit on train
    preproc, num_cols, cat_cols = build_preprocessor(train)
    X_train = preproc.fit_transform(train)
    X_test = preproc.transform(test)
    save_preprocessor(preproc, path="../models/preprocessor.joblib")

    y_train = train[target_col].values
    y_test = test[target_col].values

    if use_log:
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)
    else:
        y_train_log = y_train
        y_test_log = y_test

    results = []

    # 1) Linear regression (on log)
    lr = LinearRegression()
    lr.fit(X_train, y_train_log)
    y_pred_lr_log = lr.predict(X_test)
    if use_log:
        y_pred_lr = np.expm1(y_pred_lr_log).clip(min=0)
    else:
        y_pred_lr = y_pred_lr_log
    results.append({
        "model": "LinearRegression",
        "rmse": rmse(y_test, y_pred_lr),
        "r2": r2_score(y_test, y_pred_lr)
    })
    joblib.dump(lr, OUT_DIR / "lr_severity.joblib")

    # 2) Random Forest (quick RandomizedSearch)
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [6, 10, None],
        "min_samples_leaf": [1, 5, 10]
    }
    rf_search = RandomizedSearchCV(rf, rf_params, n_iter=6, scoring="neg_root_mean_squared_error", cv=3, random_state=random_state, n_jobs=-1)
    rf_search.fit(X_train, y_train_log)
    rf_best = rf_search.best_estimator_
    y_pred_rf_log = rf_best.predict(X_test)
    if use_log:
        y_pred_rf = np.expm1(y_pred_rf_log).clip(min=0)
    else:
        y_pred_rf = y_pred_rf_log
    results.append({
        "model": f"RandomForest_{rf_search.best_params_}",
        "rmse": rmse(y_test, y_pred_rf),
        "r2": r2_score(y_test, y_pred_rf)
    })
    joblib.dump(rf_best, OUT_DIR / "rf_severity.joblib")

    # 3) XGBoost
    xgb = XGBRegressor(tree_method="auto", n_jobs=-1, random_state=random_state, verbosity=0)
    xgb_params = {
        "n_estimators": [100, 300],
        "max_depth": [4,6,8],
        "learning_rate": [0.01, 0.1]
    }
    xgb_search = RandomizedSearchCV(xgb, xgb_params, n_iter=6, scoring="neg_root_mean_squared_error", cv=3, random_state=random_state, n_jobs=-1)
    xgb_search.fit(X_train, y_train_log)
    xgb_best = xgb_search.best_estimator_
    y_pred_xgb_log = xgb_best.predict(X_test)
    if use_log:
        y_pred_xgb = np.expm1(y_pred_xgb_log).clip(min=0)
    else:
        y_pred_xgb = y_pred_xgb_log
    results.append({
        "model": f"XGBoost_{xgb_search.best_params_}",
        "rmse": rmse(y_test, y_pred_xgb),
        "r2": r2_score(y_test, y_pred_xgb)
    })
    joblib.dump(xgb_best, OUT_DIR / "xgb_severity.joblib")

    results_df = pd.DataFrame(results).sort_values("rmse")
    results_df.to_csv(REPORT_DIR / "severity_model_comparison.csv", index=False)
    print("[+] Saved model comparison to", REPORT_DIR / "severity_model_comparison.csv")
    return results_df

if __name__ == "__main__":
    print(train_and_evaluate())
