import pandas as pd
import numpy as np
from pathlib import Path

# ================================
# Load Data
# ================================
df = pd.read_parquet("data/processed/insurance_processed.parquet")  # change if name differs

# ================================
# Cleaning & Feature Engineering
# ================================

# Filter invalid financials (documented decision)
df = df[df["TotalPremium"].notna() & df["TotalClaims"].notna()]
df = df[df["TotalPremium"] > 0]  # Exclude zero or negative premiums

# Create features
df["ClaimOccurred"] = (df["TotalClaims"] > 0).astype(int)
df["Margin"] = df["TotalPremium"] - df["TotalClaims"]


# ================================
# Select analysis variables
# ================================
cols = [
    "PolicyID", "TransactionMonth", "Province", "PostalCode", "Gender",
    "TotalPremium", "TotalClaims", "ClaimOccurred", "Margin",
    "VehicleType", "RegistrationYear", "make", "Model"
]

analysis_table = df[cols].copy()

# ================================
# Save Final Table
# ================================
output_path = Path("data/processed/analysis_table.parquet")
analysis_table.to_parquet(output_path, index=False)
print(f"Saved: {output_path}")

# ================================
# Summary Counts for Reporting
# ================================
summary = analysis_table.groupby(["Province", "PostalCode", "Gender"]).agg(
    Policies=("PolicyID", "nunique"),
    PoliciesWithClaim=("ClaimOccurred", "sum"),
    ClaimFrequency=("ClaimOccurred", "mean"),
    AvgSeverity=("TotalClaims", lambda x: np.mean(x[x>0]) if (x>0).any() else 0),
    AvgMargin=("Margin", "mean")
).reset_index()

Path("reports/task3").mkdir(parents=True, exist_ok=True)
summary_path = "reports/task3/summary_counts.csv"
summary.to_csv(summary_path, index=False)
print(f"Saved summary: {summary_path}")
