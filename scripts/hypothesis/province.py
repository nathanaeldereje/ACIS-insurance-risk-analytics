# scripts/hypothesis/province.py
"""
Hypothesis 1 (Provinces) testing utilities.

1) Frequency: Chi-square on Province x ClaimOccurred, Cramer's V, pairwise proportion z-tests (BH).
2) Severity: Conditional on ClaimOccurred==1; log transform check; Levene; ANOVA or Kruskal-Wallis;
   post-hoc: Tukey HSD or Dunn/pairwise MWU (BH).

Saves results to:
 - reports/task3/hypothesis1_frequency_summary.csv
 - reports/task3/hypothesis1_frequency_pairwise.csv
 - reports/task3/hypothesis1_severity_summary.csv
 - reports/task3/hypothesis1_severity_pairwise.csv
 - reports/task3/figures/* (png)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.api as sm
from statsmodels.stats import proportion
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import Table2x2
from scipy.stats import chi2_contingency

# PARENT_OUT_DIR = OUT_DIR.parent           # → reports/

# Optional packages: Tukey HSD and Dunn posthoc
try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
except Exception:
    pairwise_tukeyhsd = None

try:
    import scikit_posthocs as sp
except Exception:
    sp = None

sns.set(style="whitegrid")
OUT_DIR = Path("../reports/task3")
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Helper functions
# ----------------------------
def cramers_v_from_chi2(chi2, n, r, k):
    """Cramer's V for contingency table with chi2 statistic."""
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def save_df(df, fname):
    """
    Saves the DataFrame one folder **up** from the original OUT_DIR.
    """
    p = OUT_DIR / fname               # ← goes one level higher
    p.parent.mkdir(parents=True, exist_ok=True)   # ensure folder exists
    df.to_csv(p, index=False)
    print(f"[+] saved: {p}")


# ----------------------------
# 1A. Frequency: Chi-square & pairwise proportions
# ----------------------------
def test_frequency_by_province(df, province_col="Province", claim_col="ClaimOccurred",
                               min_count_per_province=50, alpha=0.05):
    """
    Perform chi-square across provinces and pairwise proportion z-tests for ClaimOccurred.

    Returns dict with keys:
      - contingency: DataFrame of counts
      - chi2_stat, chi2_p, dof, expected
      - cramer_v
      - pairwise_results: DataFrame of pairwise comparisons with p_adj (BH) and deltas + CIs
    """
    # Build contingency table
    ct = pd.crosstab(df[province_col], df[claim_col])
    # Ensure columns [0,1] exist
    for c in [0, 1]:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[[0, 1]]  # No Claim, Claim

    # Filter low-volume provinces (document decision)
    totals = ct.sum(axis=1)
    keep = totals[totals >= min_count_per_province].index
    ct_f = ct.loc[keep].copy()

    # Chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(ct_f.values)
    n = ct_f.values.sum()
    cramer_v = cramers_v_from_chi2(chi2, n, r=ct_f.shape[0], k=ct_f.shape[1])

    # Pairwise proportion z-tests between provinces
    provinces = ct_f.index.tolist()
    results = []
    for i in range(len(provinces)):
        for j in range(i+1, len(provinces)):
            a = provinces[i]; b = provinces[j]
            count = np.array([ct_f.loc[a, 1], ct_f.loc[b, 1]])
            nobs = np.array([ct_f.loc[a].sum(), ct_f.loc[b].sum()])

            # proportions_ztest from statsmodels
            stat, pval = proportion.proportions_ztest(count, nobs)
            p1 = count[0] / nobs[0]
            p2 = count[1] / nobs[1]
            delta = p1 - p2

            # 95% CI for difference using normal approximation
            se = np.sqrt(p1*(1-p1)/nobs[0] + p2*(1-p2)/nobs[1])
            z = stats.norm.ppf(1-0.025)
            ci_low = delta - z*se
            ci_high = delta + z*se

            results.append({
                "group_a": a, "group_b": b,
                "p1": p1, "p2": p2, "delta": delta,
                "z_stat": stat, "pval_raw": pval,
                "ci_low": ci_low, "ci_high": ci_high,
                "n_a": nobs[0], "n_b": nobs[1]
            })

    pairwise_df = pd.DataFrame(results)
    # multiple testing correction (Benjamini-Hochberg)
    if not pairwise_df.empty:
        reject, pvals_corrected, _, _ = multipletests(pairwise_df["pval_raw"].values, alpha=alpha, method="fdr_bh")
        pairwise_df["pval_adj"] = pvals_corrected
        pairwise_df["reject_H0"] = reject

    # Save contingency and pairwise results + summary
    save_df(ct_f.reset_index().rename(columns={0: "NoClaim", 1: "Claim"}), "hypothesis1_freq_contingency.csv")
    freq_summary = pd.DataFrame([{
        "chi2": chi2, "p_value": p, "dof": dof, "n": n, "cramers_v": cramer_v,
        "n_provinces_tested": ct_f.shape[0], "min_count_threshold": min_count_per_province
    }])
    save_df(freq_summary, "hypothesis1_frequency_summary.csv")
    if not pairwise_df.empty:
        save_df(pairwise_df, "hypothesis1_frequency_pairwise.csv")
    else:
        print("[!] No pairwise tests (not enough provinces after filtering).")

    # Plot: frequency by province with 95% CIs
    freq = ct_f[1].div(ct_f.sum(axis=1)).reset_index(name="freq")
    freq["n"] = ct_f.sum(axis=1).values
    freq["se"] = np.sqrt(freq["freq"] * (1 - freq["freq"]) / freq["n"])
    z95 = stats.norm.ppf(0.975)
    freq["ci_low"] = freq["freq"] - z95 * freq["se"]
    freq["ci_high"] = freq["freq"] + z95 * freq["se"]

    plt.figure(figsize=(10, 6))
    sns.barplot(data=freq.sort_values("freq", ascending=False), x="freq", y=province_col, palette="Blues_r")
    for i, row in freq.sort_values("freq", ascending=False).iterrows():
        plt.plot([row["ci_low"], row["ci_high"]], [i, i], color="k", lw=2)
    plt.xlabel("Claim Frequency")
    plt.title("Claim Frequency by Province (95% CI)")
    plt.tight_layout()
    fp = FIG_DIR / "hypothesis1_freq_by_province.png"
    plt.savefig(fp, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"[+] Figure saved: {fp}")

    return {
        "contingency": ct_f,
        "chi2": chi2, "p": p, "dof": dof, "expected": expected,
        "cramers_v": cramer_v,
        "pairwise": pairwise_df
    }


# ----------------------------
# 1B. Severity: conditional on ClaimOccurred == 1
# ----------------------------
def test_severity_by_province(df, province_col="Province", claim_col="ClaimOccurred",
                              severity_col="TotalClaims", min_count_per_province=30, alpha=0.05):
    """
    Test whether claim severity differs by province among policies with claims.
    Returns dict with test results, group summaries and pairwise results.
    """
    # Subset to claimants
    df_claims = df[df[claim_col] == 1].copy()
    # Keep provinces with enough claim observations
    counts = df_claims.groupby(province_col)[severity_col].count()
    keep = counts[counts >= min_count_per_province].index.tolist()
    dfk = df_claims[df_claims[province_col].isin(keep)].copy()

    # Summaries per province
    summary = dfk.groupby(province_col)[severity_col].agg(
        n="count", mean="mean", median="median", std="std", skew="skew"
    ).reset_index()
    save_df(summary, "hypothesis1_severity_summary_groups.csv")

    # Log-transform check
    dfk["log_sev"] = np.log1p(dfk[severity_col])

    # Check homogeneity of variances (Levene) on log scale
    groups_log = [g["log_sev"].values for _, g in dfk.groupby(province_col)]
    levene_stat, levene_p = stats.levene(*groups_log, center="median")
    print(f"Levene (log_sev): stat={levene_stat:.4f}, p={levene_p:.4g}")

    result = {"n_groups": len(keep), "levene_stat": levene_stat, "levene_p": levene_p}

    # Check approx normality visually/skew: we'll prefer log scale and choose test accordingly
    # If levene_p > alpha and groups not extremely skewed -> ANOVA; else Kruskal-Wallis
    # We'll compute skew per group and use threshold skew < 2 as "ok"
    skews = summary["skew"].abs().fillna(0)
    skew_ok = (skews < 2).all()
    use_anova = (levene_p > alpha) and skew_ok

    if use_anova:
        # Perform one-way ANOVA on log_sev
        groups_for_anova = [g["log_sev"].values for _, g in dfk.groupby(province_col)]
        anova_f, anova_p = stats.f_oneway(*groups_for_anova)
        result.update({"test": "ANOVA (on log)", "anova_f": anova_f, "anova_p": anova_p})
        # Post-hoc Tukey
        if pairwise_tukeyhsd is not None:
            tuk = pairwise_tukeyhsd(endog=dfk["log_sev"], groups=dfk[province_col], alpha=alpha)
            tuk_df = pd.DataFrame(data=tuk._results_table.data[1:], columns=tuk._results_table.data[0])
            save_df(tuk_df, "hypothesis1_severity_tukey.csv")
            result["posthoc"] = "tukey"
        else:
            print("[!] statsmodels.tukey not available; skipping Tukey post-hoc. Consider installing statsmodels.")
            tuk_df = pd.DataFrame()
    else:
        # Use Kruskal-Wallis
        groups_for_kw = [g[severity_col].values for _, g in dfk.groupby(province_col)]
        kw_stat, kw_p = stats.kruskal(*groups_for_kw)
        result.update({"test": "Kruskal-Wallis", "kw_stat": kw_stat, "kw_p": kw_p})
        # Post-hoc Dunn if scikit-posthocs is available, else pairwise Mann-Whitney with BH
        if sp is not None:
            dunn = sp.posthoc_dunn(dfk, val_col=severity_col, group_col=province_col, p_adjust="fdr_bh")
            dunn.reset_index().to_csv(OUT_DIR / "hypothesis1_severity_dunn_matrix.csv", index=False)
            result["posthoc"] = "dunn"
            tuk_df = pd.DataFrame()  # no tukey
        else:
            # pairwise MWU (two-sided) with BH correction
            provinces = dfk[province_col].unique()
            rows = []
            for i in range(len(provinces)):
                for j in range(i+1, len(provinces)):
                    a = provinces[i]; b = provinces[j]
                    stat, pval = stats.mannwhitneyu(dfk.loc[dfk[province_col]==a, severity_col],
                                                   dfk.loc[dfk[province_col]==b, severity_col],
                                                   alternative='two-sided')
                    rows.append({"group_a": a, "group_b": b, "pval_raw": pval})
            pairwise = pd.DataFrame(rows)
            if not pairwise.empty:
                rej, p_adj, _, _ = multipletests(pairwise["pval_raw"].values, method="fdr_bh", alpha=alpha)
                pairwise["pval_adj"] = p_adj
                pairwise["reject_H0"] = rej
                save_df(pairwise, "hypothesis1_severity_pairwise_mwu.csv")
            tuk_df = pairwise

    # Save main summary and selected results
    save_df(summary, "hypothesis1_severity_summary.csv")
    if "posthoc" in result and result["posthoc"] == "tukey" and 'tuk_df' in locals():
        save_df(tuk_df, "hypothesis1_severity_tukey.csv")
    print("[+] Severity test result:", result)
    # Plot severity boxplots (log and raw)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=province_col, y=severity_col, data=dfk)
    plt.xticks(rotation=45)
    plt.title("Claim Severity by Province (raw) — filtered provinces")
    plt.tight_layout()
    p1 = FIG_DIR / "hypothesis1_severity_box_raw.png"
    plt.savefig(p1, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"[+] Figure saved: {p1}")

    plt.figure(figsize=(12, 6))
    sns.boxplot(x=province_col, y="log_sev", data=dfk)
    plt.xticks(rotation=45)
    plt.title("Claim Severity by Province (log1p)")
    plt.tight_layout()
    p2 = FIG_DIR / "hypothesis1_severity_box_log.png"
    plt.savefig(p2, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"[+] Figure saved: {p2}")

    return {"summary": summary, "test_result": result, "pairwise": tuk_df if 'tuk_df' in locals() else None}


# ----------------------------
# Usage example (run in notebook or script)
# ----------------------------
if __name__ == "__main__":
    # Assume you have an analysis table parquet created earlier (one row per policy)
    # e.g., data/processed/analysis_table.parquet
    analysis_path = Path("data/processed/analysis_table.parquet")
    if not analysis_path.exists():
        raise FileNotFoundError(f"Run task3_prepare_analysis_table.py first. Missing: {analysis_path}")

    df = pd.read_parquet(analysis_path)

    # Ensure ClaimOccurred is 0/1
    if "ClaimOccurred" not in df.columns:
        df["ClaimOccurred"] = (df["TotalClaims"] > 0).astype(int)

    # Frequency test
    freq_res = test_frequency_by_province(df, min_count_per_province=100)
    # Severity test
    sev_res = test_severity_by_province(df, min_count_per_province=30)

    print("Done. Outputs in reports/task3/ and figures in reports/task3/figures/")
