# scripts/07_shap_severity.py
import joblib
import shap
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = "../models/xgb_severity.joblib"
PREPROC_PATH = "../models/preprocessor.joblib"
REPORT_DIR = Path("../reports/task4")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def shap_for_best_model(sample_path="../data/processed/severity_test.parquet", n_samples=2000):
    model = joblib.load(MODEL_PATH)
    preproc = joblib.load(PREPROC_PATH)
    df = pd.read_parquet(sample_path)
    X = preproc.transform(df)
    # choose a sample to speed up
    idx = np.random.choice(X.shape[0], min(n_samples, X.shape[0]), replace=False)
    Xs = X[idx]
    explainer = shap.Explainer(model)
    shap_values = explainer(Xs)
    # summary plot (save)
    plt.figure()
    shap.summary_plot(shap_values, features=Xs, show=False)
    plt.savefig(REPORT_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    # mean abs shap per feature
    shap_df = pd.DataFrame(np.abs(shap_values.values).mean(axis=0), columns=["mean_abs_shap"])
    # we need column names from the preprocessor: reconstruct them
    # quick way: combine numeric and categorical names
    # NOTE: ensure preprocessor stores feature names or rebuild using train metadata
    # Save shap importance
    shap_df.to_csv(REPORT_DIR / "shap_feature_importance.csv", index=False)
    print("[+] SHAP outputs saved to", REPORT_DIR)
