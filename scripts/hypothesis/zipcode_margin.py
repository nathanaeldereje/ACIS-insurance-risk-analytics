# scripts/hypothesis/zipcode_margin.py
"""
Hypothesis 3: Margin Differences by Postal Code

H₀ = no significant difference in policy margin between ZIP codes.

Margin per policy = TotalPremium - TotalClaims
Test approach: Linear regression (OLS) with robust SE or ANOVA/Kruskal-Wallis if simpler.
Outputs saved to: reports/task3/hypothesis3/
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def test_margin_by_zip(
    df: pd.DataFrame,
    zip_col: str = "PostalCode",
    premium_col: str = "TotalPremium",
    claims_col: str = "TotalClaims",
    min_policies: int = 50,
    reports_folder: str = "../reports/task3/hypothesis3"
) -> dict:
    """
    Test Hypothesis 3: Whether insurance policy margins differ significantly
    across ZIP/postal codes.

    This function computes policy-level margins, filters ZIP codes with a minimum
    number of policies, performs descriptive summary statistics, fits an OLS model 
    with cluster-robust standard errors by ZIP, performs a Wald test on the ZIP 
    fixed effects, and generates visualizations for the top and bottom ZIPs.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing at least the premium, claims, and ZIP code columns.
    
    zip_col : str, default "PostalCode"
        Column name for ZIP / postal codes.

    premium_col : str, default "TotalPremium"
        Column name representing the total premium per policy.

    claims_col : str, default "TotalClaims"
        Column name representing the total claims per policy.

    min_policies : int, default 50
        Minimum number of policies required for a ZIP code to be included in the analysis.
        ZIP codes with fewer observations are removed.

    reports_folder : str, default "../reports/task3/hypothesis3"
        Path to the folder where summary tables and plots will be saved.

    Returns
    -------
    dict
        A dictionary containing:
        
        - **p_value** : float  
            P-value for the overall ZIP code effect from the robust Wald test.
        
        - **summary** : pandas.DataFrame  
            Summary statistics (mean margin, std, confidence intervals) per ZIP code.
        
        - **model** : statsmodels.regression.linear_model.RegressionResultsWrapper  
            Fitted OLS model object with clustered standard errors.

    Notes
    -----
    - Margin is computed as: ``TotalPremium - TotalClaims``.
    - OLS model: ``Margin ~ C(ZipCode)`` (ZIP fixed effects).
    - Clustered standard errors are used to account for intra-ZIP correlation.
    - Produces two plots:
        * Top 10 ZIPs by mean margin
        * Bottom 10 ZIPs by mean margin
    - Saves summary table as CSV in the reports folder.

    Examples
    --------
    >>> results = test_margin_by_zip(df, "ZIP", "Premium", "Claims")
    >>> print(results["p_value"])
    >>> results["summary"].head()

    """

    _ensure_dir(reports_folder)
    _ensure_dir(f"{reports_folder}/figures")

    df = df.copy()
    df["Margin"] = df[premium_col] - df[claims_col]

    # Filter ZIPs with enough policies
    zip_counts = df.groupby(zip_col).size()
    eligible = zip_counts[zip_counts >= min_policies].index
    df_filtered = df[df[zip_col].isin(eligible)].copy()

    # Summary statistics
    zip_summary = df_filtered.groupby(zip_col)["Margin"].agg(
        n="count", mean="mean", std="std"
    ).sort_values("mean", ascending=False)

    zip_summary["ci_lower"] = zip_summary["mean"] - 1.96 * zip_summary["std"] / np.sqrt(zip_summary["n"])
    zip_summary["ci_upper"] = zip_summary["mean"] + 1.96 * zip_summary["std"] / np.sqrt(zip_summary["n"])

    zip_summary.to_csv(f"{reports_folder}/zip_margin_summary.csv")

    # Cluster-robust OLS
    formula = f"Margin ~ C({zip_col})"
    model = smf.ols(formula, data=df_filtered).fit(
        cov_type="cluster", cov_kwds={"groups": df_filtered[zip_col]}
    )

    # Robust Wald test for ZIP effect
    wald_table = model.wald_test_terms().table
    print(wald_table)  # check actual column names

    # Attempt to extract p-value using likely column names
    if "P>F" in wald_table.columns:
        zip_p = wald_table.loc[f"C({zip_col})", "P>F"]
    elif "Pr > F" in wald_table.columns:
        zip_p = wald_table.loc[f"C({zip_col})", "Pr > F"]
    elif "pvalue" in wald_table.columns:
        zip_p = wald_table.loc[f"C({zip_col})", "pvalue"]
    else:
        raise KeyError(f"Cannot find p-value column in Wald test table:\n{wald_table.columns}")


    # Top and bottom ZIPs
    top_zip = zip_summary.head(10)
    bottom_zip = zip_summary.tail(10)

    # Plot helpers
    def _plot_zip_bar(data, title, path, color):
        yerr = np.vstack([
            data["mean"] - data["ci_lower"],
            data["ci_upper"] - data["mean"]
        ])
        plt.figure(figsize=(12, 6))
        plt.bar(data.index, data["mean"], yerr=yerr, alpha=0.7, capsize=5, color=color)
        plt.ylabel("Mean Margin")
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()

    _plot_zip_bar(top_zip, "Top 10 ZIP Codes by Mean Margin",
                  f"{reports_folder}/figures/top10_zip_margin.png", "green")

    _plot_zip_bar(bottom_zip, "Bottom 10 ZIP Codes by Mean Margin",
                  f"{reports_folder}/figures/bottom10_zip_margin.png", "red")

    print(f"H3 Margin → ZIP effect p = {zip_p:.2e}")

    return {
        "p_value": zip_p,
        "summary": zip_summary,
        "model": model
    }
