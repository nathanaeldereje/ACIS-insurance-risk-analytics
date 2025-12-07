"""
EDA METRICS MODULE: Loss Ratio & Portfolio Analysis
---------------------------------------------------
Usage in notebook:

    from scripts.eda_metrics import portfolio_loss_ratio, loss_ratio_by
    
    portfolio_loss_ratio(df)
    loss_ratio_by(df, "Province", min_transactions=1000)
"""

import numpy as np
import pandas as pd


# =========================================================
# 1) SAFE LOSS RATIO FUNCTION
# =========================================================
def safe_loss_ratio(claims, premium):
    """
    Works with scalars OR pandas Series.
    Prevents division by zero and returns clean loss ratio.

    Parameters
    ----------
    claims : float or numeric
    premium : float or numeric

    Returns
    -------
    float or NaN
    """
    try:
        claims = float(claims)
        premium = float(premium)
    except (ValueError, TypeError):
        return np.nan

    if premium == 0 or premium is None or np.isnan(premium):
        return np.nan

    return round(claims / premium, 5)


# =========================================================
# 2) OVERALL PORTFOLIO LOSS RATIO
# =========================================================
def portfolio_loss_ratio(df: pd.DataFrame, premium_col="TotalPremium", claims_col="TotalClaims"):
    """
    Computes and prints overall portfolio loss ratio.

    Returns
    -------
    dict
        {"TotalPremium": value, "TotalClaims": value, "LossRatio": value}
    """
    total_premium = df[premium_col].sum()
    total_claims = df[claims_col].sum()
    overall_lr = safe_loss_ratio(total_claims, total_premium)

    print(f"\n{'='*65}")
    print(f"OVERALL PORTFOLIO LOSS RATIO (Feb 2014 – Aug 2015)")
    print(f"{'='*65}")
    print(f"Total Premium Collected : R{total_premium:,.0f}")
    print(f"Total Claims Paid       : R{total_claims:,.0f}")
    print(f"Overall Loss Ratio      : {overall_lr:.4f} → {overall_lr:.2%}")
    print(f"{'='*65}")

    return {
        "TotalPremium": total_premium,
        "TotalClaims": total_claims,
        "LossRatio": overall_lr
    }


# =========================================================
# 3) LOSS RATIO BY GROUP (Province, Gender, VehicleType, etc.)
# =========================================================
def loss_ratio_by(
    df: pd.DataFrame,
    by_column: str,
    min_transactions: int = 500,
    premium_col="TotalPremium",
    claims_col="TotalClaims",
    id_col="UnderwrittenCoverID"
):
    """
    Group-based loss ratio calculations (Province, Gender, VehicleType, etc.)

    Returns
    -------
    pd.DataFrame : sorted by loss ratio DESC, filtered by transaction threshold.
    """
    grouped = df.groupby(by_column,observed=False).agg(
        TotalPremium=(premium_col, "sum"),
        TotalClaims=(claims_col, "sum"),
        Transactions=(id_col, "count")
    ).round(2)

    # Apply safe division
    grouped["LossRatio"] = grouped.apply(
        lambda row: safe_loss_ratio(row["TotalClaims"], row["TotalPremium"]), axis=1
    )

    # Filter by minimum transaction volume
    grouped = grouped[grouped["Transactions"] >= min_transactions]

    return grouped.sort_values("LossRatio", ascending=False).reset_index()




# =======================
# 1. FINANCIAL SUMMARY
# =======================
def describe_financials(df):
    """
    Return full statistical description for financial insurance columns.
    Includes extended percentiles, skew, zero %, and missing %.
    """
    cols = [
        'TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm',
        'SumInsured', 'CustomValueEstimate', 'CapitalOutstanding'
    ]
    
    desc = df[cols].describe(
        percentiles=[0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]
    ).T
    
    desc['skew'] = df[cols].skew()
    desc['%_zeros'] = (df[cols] == 0).mean() * 100
    desc['%_missing'] = df[cols].isna().mean() * 100
    
    return desc.round(2)

# =======================
# 2. EXTREME OUTLIERS RULES
# =======================
def extreme_claims(series, threshold=0.99):
    """
    Identify extreme claims using percentile thresholds (default = 99th).
    Standard for insurance actuarial analysis.
    """
    p = series.quantile(threshold)
    return series[series > p]

def extreme_value(series, threshold=0.995):
    """
   Identify extreme valuation exposures such as SumInsured.
   99.5% is actuarially standard for pricing and fraud segmentation.
    """
    p = series.quantile(threshold)
    return series[series > p]

# =======================
# 3. PRINT UTILITY
# =======================
def print_extremes(df):
    """
    Pretty print standardized extreme values across claims + insured values.
    """
    claims_99 = extreme_claims(df['TotalClaims'], 0.99)
    claims_999 = extreme_claims(df['TotalClaims'], 0.999)
    custom_995 = extreme_value(df['CustomValueEstimate'].dropna(), 0.995)

    print("\nEXTREME VALUES (Insurance-standard thresholds)")
    print(f"• Claims > 99th percentile   : {len(claims_99):,} policies | Max = R{df['TotalClaims'].max():,.0f}")
    print(f"• Claims > 99.9th percentile : {len(claims_999):,} policies (the real monsters)")
    print(f"• CustomValueEstimate > 99.5th: {len(custom_995):,} policies | Max = R{df['CustomValueEstimate'].max():,.0f}")


# =======================
# 3. TEMPORAL TRENDS
# =======================
def temporal_trends(df):
    """
    Computes monthly aggregated insurance KPIs:
    - Premium, Claims, Policies
    - Loss Ratio (Claims/Premium)
    - Claim Frequency (policies with ≥1 claim / total policies)
    """
    monthly = df.groupby('TransactionYearMonth').agg(
        TotalPremium=('TotalPremium', 'sum'),
        TotalClaims=('TotalClaims', 'sum'),
        Policies=('PolicyID', 'nunique')
    ).reset_index()

    monthly['LossRatio'] = monthly['TotalClaims'] / monthly['TotalPremium'].replace({0: np.nan})

    # Claim frequency calc
    claims_per_policy_month = df.groupby(
        ['TransactionYearMonth', 'PolicyID']
    )['TotalClaims'].sum().reset_index()

    claims_per_month = claims_per_policy_month.groupby(
        'TransactionYearMonth'
    ).agg(
        PoliciesWithClaim=('TotalClaims', lambda x: (x > 0).sum()),
        Policies=('TotalClaims', 'count')
    ).reset_index()

    claims_per_month['ClaimFrequency'] = (
        claims_per_month['PoliciesWithClaim'] / claims_per_month['Policies']
    )

    # Merge
    monthly = monthly.merge(
        claims_per_month[['TransactionYearMonth', 'ClaimFrequency']],
        on='TransactionYearMonth',
        how='left'
    )

    return monthly


# =======================
# 4. KPI BY VEHICLE MAKE
# =======================
def loss_ratio_by_make(df, min_policies=50):
    make_stats = df.groupby('make').agg(
        TotalPremium=('TotalPremium', 'sum'),
        TotalClaims=('TotalClaims', 'sum'),
        Count=('PolicyID', 'count')
    ).reset_index()
    make_stats['make']=make_stats['make'].str.strip()
    make_stats['LossRatio'] = (
        make_stats['TotalClaims'] / make_stats['TotalPremium'].replace({0: np.nan})
    )

    return make_stats[make_stats['Count'] >= min_policies].sort_values('LossRatio', ascending=False)


def loss_ratio_by_model(df, make, min_policies=20):
    subset = df[df['make'] == make]
    if subset.empty:
        return pd.DataFrame()

    model_stats = subset.groupby('Model',observed=False).agg(
        TotalPremium=('TotalPremium', 'sum'),
        TotalClaims=('TotalClaims', 'sum'),
        Count=('PolicyID', 'count')
    ).reset_index()

    model_stats['LossRatio'] = (
        model_stats['TotalClaims'] / model_stats['TotalPremium'].replace({0: np.nan})
    )

    return model_stats[model_stats['Count'] >= min_policies].sort_values('LossRatio', ascending=False)
