# scripts/hypothesis/gender.py
"""
Hypothesis 4: Gender Differences (Women vs Men)

Tests both:
- Claim frequency differences by gender
- Claim severity differences by gender (conditional on having a claim)

Outputs saved to: reports/task3/hypothesis4/
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def test_frequency_by_gender(
    df: pd.DataFrame,
    gender_col: str = "Gender",
    claim_col: str = "ClaimOccurred",
    reports_folder: str = "../reports/task3/hypothesis4"
) -> dict:
    """
    Test whether **claim frequency** differs between genders using a two-proportion z-test.

    Frequency is calculated as:
        Claim Frequency = (# of policies with a claim) / (total # of policies)

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing gender and claim indicator columns.

    gender_col : str, default "Gender"
        Column specifying policyholder gender (ideally two categories).

    claim_col : str, default "ClaimOccurred"
        Binary column indicating whether a claim occurred (0/1 or True/False).

    reports_folder : str, default "../reports/task3/hypothesis4"
        Directory where CSV summaries and test results will be saved.

    Returns
    -------
    dict
        Contains:
        - **p_value** : float  
            P-value from the two-proportion z-test comparing frequency between genders.
        - **effect_size_h** : float  
            Cohen’s h effect size for proportion differences.
        - **proportions** : pandas.Series  
            Claim proportions by gender.

    Notes
    -----
    - Performs a classic two-proportion z-test.
    - Computes Cohen’s h for effect size:
        h = 2 * (arcsin(sqrt(p1)) − arcsin(sqrt(p2)))
    - Saves:
        - frequency_by_gender.csv  
        - frequency_test_result.txt

    Examples
    --------
    >>> result = test_frequency_by_gender(df)
    >>> result["p_value"]
    """

    _ensure_dir(reports_folder)

    gender_counts = df.groupby(gender_col)[claim_col].agg(['sum', 'count'])
    gender_counts['prop'] = gender_counts['sum'] / gender_counts['count']

    # Two-proportion z-test
    n1, n2 = gender_counts['count']
    p1, p2 = gender_counts['prop']
    p_pool = (p1*n1 + p2*n2) / (n1+n2)
    z_stat = (p1 - p2) / np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Cohen's h
    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

    # Save results
    gender_counts.to_csv(f"{reports_folder}/frequency_by_gender.csv")
    with open(f"{reports_folder}/frequency_test_result.txt", "w") as f:
        f.write(
            f"Two-proportion z-test:\n"
            f"Z = {z_stat:.4f}, p = {p_value:.4e}, Cohen's h = {h:.4f}\n"
        )

    print(f"Frequency → p = {p_value:.2e}, Cohen's h = {h:.2f}")

    return {"p_value": p_value, "effect_size_h": h, "proportions": gender_counts['prop']}


def test_severity_by_gender(
    df: pd.DataFrame,
    gender_col: str = "Gender",
    severity_col: str = "TotalClaims",
    reports_folder: str = "reports/task3/hypothesis4"
) -> dict:
    """
    Test whether **claim severity** differs between genders.

    Severity model:
    - Only policies with a positive claim amount are included.
    - Automatically chooses:
        * **t-test** (equal variance / Welch) when appropriate
        * **Mann–Whitney U test** when sample sizes are small or variance assumptions fail

    Effect size:
    - Cohen’s d for mean differences (for any t-test scenario)

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with gender and claim severity.

    gender_col : str, default "Gender"
        Gender column name.

    severity_col : str, default "TotalClaims"
        Claim severity column (numeric, nonzero values indicate claim severity).

    reports_folder : str, default "reports/task3/hypothesis4"
        Directory where summaries and test results will be stored.

    Returns
    -------
    dict
        Contains:
        - **p_value** : float  
            P-value from the chosen statistical test.
        - **effect_size_d** : float  
            Cohen’s d effect size for severity differences.
        - **test_used** : str  
            Name of the statistical test used (“t-test”, “Welch’s”, “Mann-Whitney”).

    Notes
    -----
    - Performs Levene’s test to select between equal-variance t-test or Welch.
    - If sample sizes < 10 in any gender, switches to Mann-Whitney U.
    - Saves:
        - severity_by_gender.csv
        - severity_test_result.txt

    Examples
    --------
    >>> result = test_severity_by_gender(df)
    >>> result["test_used"], result["p_value"]
    """

    _ensure_dir(reports_folder)

    df_claims = df[df[severity_col] > 0].copy()
    groups = [g[severity_col].values for _, g in df_claims.groupby(gender_col)]
    labels = df_claims[gender_col].unique()

    # Variance test
    stat_levene, p_levene = stats.levene(*groups)
    if p_levene > 0.05:
        stat_test, p_value = stats.ttest_ind(*groups, equal_var=True)
        test_used = "t-test (equal var)"
    else:
        stat_test, p_value = stats.ttest_ind(*groups, equal_var=False)
        test_used = "Welch's t-test"

    # Use nonparametric test if sample extremely small
    if np.any([len(g) < 10 for g in groups]):
        stat_test, p_value = stats.mannwhitneyu(*groups, alternative='two-sided')
        test_used = "Mann-Whitney U"

    # Effect size: Cohen’s d
    n1, n2 = len(groups[0]), len(groups[1])
    mean1, mean2 = np.mean(groups[0]), np.mean(groups[1])
    s1, s2 = np.var(groups[0], ddof=1), np.var(groups[1], ddof=1)
    pooled_sd = np.sqrt((s1 + s2) / 2)
    cohen_d = (mean1 - mean2) / pooled_sd if pooled_sd > 0 else np.nan

    # Save results
    df_claims.groupby(gender_col)[severity_col].agg(['count', 'mean', 'median']).to_csv(
        f"{reports_folder}/severity_by_gender.csv"
    )
    with open(f"{reports_folder}/severity_test_result.txt", "w") as f:
        f.write(
            f"{test_used}:\n"
            f"Statistic = {stat_test:.4f}, p = {p_value:.4e}, Cohen's d = {cohen_d:.4f}\n"
        )

    print(f"Severity → {test_used} p = {p_value:.2e}, Cohen's d = {cohen_d:.2f}")

    return {"p_value": p_value, "effect_size_d": cohen_d, "test_used": test_used}
