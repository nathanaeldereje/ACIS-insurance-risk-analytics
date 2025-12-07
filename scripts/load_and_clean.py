"""
LOAD & CLEAN MODULE for ACIS Insurance Dataset
----------------------------------------------
Usage in notebook:
    from scripts.load_and_clean import load_insurance_data
    df = load_insurance_data()
"""

import pandas as pd
import numpy as np



# Explicit dtypes + proper parsing
DTYPE_MAP = {
    # Columns that contain mixed types
    'CapitalOutstanding': 'string',    # column 32
    'NewVehicle': 'string',            # column 37 contains Yes/No + garbage
    
    # Categorical optimization (improves grouping performance later)
    'Province': 'category',
    'PostalCode': 'category',
    'Gender': 'category',
    'VehicleType': 'category',
    'CoverCategory': 'category',
    'CoverType': 'category',
    'CoverGroup': 'category',
    'make': 'category',
    'Model': 'category',
    'bodytype': 'category',
}

# Parse as date
PARSE_DATES = ['TransactionMonth']


# =========================================================
# MAIN LOADER FUNCTION
# =========================================================
def load_insurance_data(path) -> pd.DataFrame:
    """
    Load and clean the ACIS insurance dataset with optimized dtype handling.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset ready for EDA and modeling.
    """
    
    # === LOAD ===
    df = pd.read_csv(
        path,
        sep="|",
        dtype=DTYPE_MAP,
        parse_dates=PARSE_DATES,
        low_memory=False,
        na_values=["", "NULL", "NaN"]
    )
    
    # === QUICK FIXES ===
    
    # Convert CapitalOutstanding into numeric (invalid -> NaN)
    df["CapitalOutstanding"] = pd.to_numeric(df["CapitalOutstanding"], errors="coerce")
    
    # Clean NewVehicle → True / False / <NA>
    df["NewVehicle"] = (
        df["NewVehicle"]
        .astype("string")
        .str.strip()
        .str.title()
        .map({"Yes": True, "No": False})
        .astype("boolean")  # Nullable boolean type
    )
    df["make"]=df['make'].str.strip()
    # Extract Period (Year-Month)
    df["TransactionYearMonth"] = df["TransactionMonth"].dt.to_period("M")
    
    # OPTIONAL: return optimization summary
    print("=" * 55)
    print("DATA LOADED SUCCESSFULLY!")
    print("=" * 55)
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print("=" * 55)
    
    return df

