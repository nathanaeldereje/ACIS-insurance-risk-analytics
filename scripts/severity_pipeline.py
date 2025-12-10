# scripts/severity_pipeline.py
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

def build_preprocessor(df, categorical_thresh=50):
    # pick columns
    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    # remove target and IDs
    for drop in ["TotalClaims","PolicyID","UnderwrittenCoverID","TransactionMonth","TransactionYearMonth"]:
        if drop in numeric_cols:
            numeric_cols.remove(drop)
    # pick categorical
    categorical_cols = df.select_dtypes(include=["category","object"]).columns.tolist()
    # keep only categories with reasonable cardinality (or leave; OHE on high-cardinals -> use target encoding later)
    # for now we'll OneHot up to moderate cardinality, else drop or label encode
    # Build transformations
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )
    return preprocessor, numeric_cols, categorical_cols

def save_preprocessor(preprocessor, path="../models/preprocessor.joblib"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"[+] saved preprocessor to {path}")
