# scripts/prepare_severity_data.py
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

OUT_DIR = Path("../data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def prepare_severity_dataset(
    analysis_table_path: str = "../data/processed/analysis_table.parquet",
    target_col: str = "TotalClaims",
    min_amount: float = 0.0,
    test_size: float = 0.2,
    random_state: int = 42,
    save_prefix: str = "severity"
):
    """
    Prepare dataset for claim severity modeling:
      - Filter rows where TotalClaims > min_amount
      - Create features (vehicle_age, claim_flag already in analysis table)
      - Handle simple missing-value placeholders (we'll keep advanced imputation for modeling step)
      - Train/test split (stratify by year-month or by large categorical if needed)
    Saves train/test parquet to data/processed/{save_prefix}_train.parquet and {...}_test.parquet
    Returns: train_df, test_df
    """
    df = pd.read_parquet(analysis_table_path)
    # Filter to claimants (severity model)
    df = df[df[target_col] > min_amount].copy()
    # Basic feature engineering
    # vehicle age if RegistrationYear exists and TransactionMonth exists
    if "RegistrationYear" in df.columns and "TransactionMonth" in df.columns:
        df["VehicleAge"] = df["TransactionMonth"].dt.year - df["RegistrationYear"]
        df["VehicleAge"] = df["VehicleAge"].clip(lower=0)
    # Example: ratio features
    df["ClaimToPremiumRatio"] = df[target_col] / (df["TotalPremium"].replace({0: np.nan}))
    # Drop rows with missing target (shouldn't be any) or missing key id
    df = df.dropna(subset=[target_col])
    # train-test split by policy (to avoid leakage), stratify by TransactionYearMonth if many months
    if "TransactionYearMonth" in df.columns:
        stratify = df["TransactionYearMonth"].astype(str)
    else:
        stratify = None
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)
    train_path = OUT_DIR / f"{save_prefix}_train.parquet"
    test_path = OUT_DIR / f"{save_prefix}_test.parquet"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    print(f"[+] saved {train.shape[0]} train rows → {train_path}")
    print(f"[+] saved {test.shape[0]} test rows  → {test_path}")
    return train, test

if __name__ == "__main__":
    prepare_severity_dataset()
