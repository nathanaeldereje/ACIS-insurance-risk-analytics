"""
scripts/visualizations.py

Plotting utilities for ACIS-insurance-risk-analytics.

Each plotting function displays the figure (for notebooks) and optionally saves
a high-resolution PNG into `reports/figures/`.

Usage (notebook):
    from scripts.visualizations import (
        plot_financial_distributions,
        plot_loss_ratio_by,
        plot_monthly_trends,
        plot_top_risky_makes,
        plot_correlation,
        plot_log_transforms
    )

    plot_financial_distributions(df)
    agg = plot_loss_ratio_by(df, "Province", min_transactions=1000)
    plot_monthly_trends(monthly)
    plot_top_risky_makes(makes)
    plot_correlation(df, ['TotalPremium','TotalClaims','SumInsured','CapitalOutstanding'])
    plot_log_transforms(df)
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


# ---------------------------------------------------------------------
# Helper: Save figure to reports/figures
# ---------------------------------------------------------------------
def _save_fig(filename: str, out_dir: str = "reports/figures") -> str:
    """
    Save the current matplotlib figure to disk.

    Parameters
    ----------
    filename : str
        File name for the saved figure (e.g., "monthly_trends.png").
    out_dir : str, optional
        Directory where the figure will be saved. Default: "reports/figures".

    Returns
    -------
    str
        The full path to the saved file.

    Side effects
    ------------
    - Creates the output directory if it does not exist.
    - Saves the current active matplotlib figure with dpi=300 and bbox_inches='tight'.
    - Prints a short confirmation message.
    """
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, filename)
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    print(f"[+] Figure saved: {file_path}")
    return file_path


# ---------------------------------------------------------------------
# 1. Financial distributions (Histogram + Boxplot)
# ---------------------------------------------------------------------
def plot_financial_distributions(df: pd.DataFrame, save: bool = True) -> None:
    """
    Plot histograms (log scale on y-axis) and boxplots (up to 99th percentile)
    for a small set of key financial variables.

    Parameters
    ----------
    df : pandas.DataFrame
        The insurance dataframe with columns:
        - 'TotalPremium', 'TotalClaims', 'SumInsured', 'CapitalOutstanding'
    save : bool, optional
        If True, save the combined figure to `reports/figures/financial_distributions.png`.
        Default is True.

    Returns
    -------
    None
        The function displays the figure and optionally saves it.

    Notes
    -----
    - Uses log scaling for the histogram y-axis to make heavy tails readable.
    - The boxplots are clipped at the 99th percentile to make the main mass visible.

    Example
    -------
    >>> plot_financial_distributions(df)
    """
    cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CapitalOutstanding']
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    fig.suptitle('Key Financial Variables – Insurance-Grade Visualizations',
                 fontsize=20, fontweight='bold', y=1.02)

    for i, col in enumerate(cols):
        data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        positive = data[data > 0]

        ax_hist = axes[i]
        sns.histplot(positive, bins=80, kde=True, color=colors[i],
                     alpha=0.7, ax=ax_hist, log_scale=(False, True))
        ax_hist.set_title(f'{col} Distribution (Log Scale)', fontsize=14, fontweight='bold')
        ax_hist.set_xlabel(f'{col} (R)')
        ax_hist.set_ylabel('Frequency (log)')

        ax_box = axes[i + 4]
        q99 = data.quantile(0.99)
        box_data = data[data <= q99]
        sns.boxplot(x=box_data, color=colors[i], ax=ax_box, width=0.4)
        ax_box.set_title(f'{col} – Boxplot (up to 99th %ile)', fontsize=14, fontweight='bold')
        ax_box.set_xlabel('')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    if save:
        _save_fig("financial_distributions.png")
    plt.show()


# ---------------------------------------------------------------------
# 2. Loss ratio by category (Province, Gender, VehicleType, etc.)
# ---------------------------------------------------------------------
def plot_loss_ratio_by(
    df: pd.DataFrame,
    column: str,
    min_transactions: int = 500,
    title: str | None = None,
    palette: str = 'Reds_r',
    save: bool = True
) -> pd.DataFrame:
    """
    Compute loss-ratio by a categorical column and plot a horizontal bar chart.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe that contains 'TotalPremium', 'TotalClaims', and an ID column.
    column : str
        The categorical column to group by (e.g., 'Province', 'VehicleType', 'Gender').
    min_transactions : int, optional
        Minimum number of transactions required for a group to be included.
        Default is 500.
    title : str or None, optional
        Custom title for the plot. If None, a default title is used.
    palette : str, optional
        Seaborn palette name for the bars. Default 'Reds_r'.
    save : bool, optional
        If True, save the plot to `reports/figures/loss_ratio_by_{column}.png`.

    Returns
    -------
    pandas.DataFrame
        Aggregated table with columns: [column, TotalPremium, TotalClaims, Transactions, LossRatio]
        The returned DataFrame is sorted by LossRatio descending and includes only groups
        meeting min_transactions.

    Side effects
    ------------
    Displays a bar plot and optionally saves it.

    Example
    -------
    >>> prov_df = plot_loss_ratio_by(df, "Province", min_transactions=1000)
    """
    agg = (df.groupby(column, observed=False)
             .agg(TotalPremium=('TotalPremium', 'sum'),
                  TotalClaims=('TotalClaims', 'sum'),
                  Transactions=('UnderwrittenCoverID', 'count'))
             .assign(LossRatio=lambda x: x.TotalClaims / x.TotalPremium.replace(0, np.nan))
             .query(f'Transactions >= {min_transactions}')
             .sort_values('LossRatio', ascending=False)
             .reset_index())

    plt.figure(figsize=(12, 6))
    sns.barplot(data=agg, x='LossRatio', y=column, hue=column,
                palette=palette, legend=False, edgecolor='black')

    plt.title(title or f'Loss Ratio by {column}', fontsize=16, fontweight='bold')
    plt.xlabel('Loss Ratio')
    plt.ylabel('')
    plt.tight_layout()

    if save:
        safe_name = f"loss_ratio_by_{column.lower().replace(' ', '_')}.png"
        _save_fig(safe_name)

    plt.show()
    return agg


# ---------------------------------------------------------------------
# 3. Temporal trends (monthly LossRatio + ClaimFrequency)
# ---------------------------------------------------------------------
def plot_monthly_trends(monthly_df: pd.DataFrame, save: bool = True) -> None:
    """
    Plot monthly LossRatio and ClaimFrequency on a dual-axis time series.

    Parameters
    ----------
    monthly_df : pandas.DataFrame
        Dataframe produced by temporal_trends(...) with columns:
        ['TransactionYearMonth', 'TotalPremium', 'TotalClaims', 'Policies', 'LossRatio', 'ClaimFrequency'].
    save : bool, optional
        If True, save the plot to `reports/figures/monthly_trends.png`.

    Returns
    -------
    None

    Example
    -------
    >>> monthly = temporal_trends(df)
    >>> plot_monthly_trends(monthly)
    """
    monthly_df = monthly_df.copy()
    monthly_df['Date'] = monthly_df['TransactionYearMonth'].dt.to_timestamp()
    monthly_df = monthly_df[monthly_df['Policies'] > 500]

    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()

    sns.lineplot(data=monthly_df, x='Date', y='LossRatio',
                 marker='o', color='#d62728', linewidth=3, markersize=8, label='Loss Ratio', ax=ax1)

    ax2 = ax1.twinx()
    sns.lineplot(data=monthly_df, x='Date', y='ClaimFrequency',
                 marker='X', color='#1f77b4', linewidth=3, markersize=10, label='Claim Frequency', ax=ax2)

    plt.title('Temporal Trends: Loss Ratio & Claim Frequency', fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Loss Ratio', color='#d62728', fontweight='bold')
    ax2.set_ylabel('Claim Frequency (%)', color='#1f77b4', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        _save_fig("monthly_trends.png")
    plt.show()


# ---------------------------------------------------------------------
# 4. Top risky makes bar chart
# ---------------------------------------------------------------------
def plot_top_risky_makes(makes_df: pd.DataFrame, top_n: int = 5, save: bool = True) -> None:
    """
    Plot the top N riskiest vehicle makes by LossRatio.

    Parameters
    ----------
    makes_df : pandas.DataFrame
        Dataframe produced by loss_ratio_by_make(...) with columns:
        ['make', 'TotalPremium', 'TotalClaims', 'Count', 'LossRatio'].
    top_n : int, optional
        Number of top makes to show. Default is 5.
    save : bool, optional
        If True, save the plot to `reports/figures/top_risky_makes.png`.

    Returns
    -------
    None

    Example
    -------
    >>> makes = loss_ratio_by_make(df, min_policies=50)
    >>> plot_top_risky_makes(makes, top_n=10)
    """
    top = makes_df.sort_values('LossRatio', ascending=False).head(top_n).copy()

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(data=top, y='make', x='LossRatio', hue='make',
                     palette='Reds_r', order=top['make'],
                     legend=False, edgecolor='black', linewidth=1.5)

    for i, ratio in enumerate(top['LossRatio']):
        ax.text(ratio + 0.25, i, f'{ratio:.2f}', va='center',
                fontweight='bold', fontsize=11, color='darkred')

    # Emphasize the top bar visually if it exists
    if ax.patches:
        ax.patches[0].set_facecolor('#8B0000')
        ax.patches[0].set_edgecolor('black')
        ax.patches[0].set_linewidth(4)

    plt.title(f'Top {top_n} Riskiest Vehicle Makes', fontsize=17, fontweight='bold')
    plt.xlabel('Loss Ratio (Claims ÷ Premium)')
    plt.ylabel('')
    plt.tight_layout()

    if save:
        _save_fig("top_risky_makes.png")
    plt.show()


# ---------------------------------------------------------------------
# 5. Correlation heatmap
# ---------------------------------------------------------------------
def plot_correlation(df: pd.DataFrame, cols: list[str], save: bool = True) -> None:
    """
    Plot a correlation heatmap for the given numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe with numeric columns.
    cols : list[str]
        List of column names to include in the correlation matrix.
    save : bool, optional
        If True, save the plot to `reports/figures/correlation_matrix.png`.

    Returns
    -------
    None

    Example
    -------
    >>> plot_correlation(df, ['TotalPremium','TotalClaims','SumInsured'])
    """
    plt.figure()
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Key Financial Variables')
    if save:
        _save_fig("correlation_matrix.png")
    plt.show()


# ---------------------------------------------------------------------
# 6. Log-transform insight plots
# ---------------------------------------------------------------------
def plot_log_transforms(df: pd.DataFrame, save: bool = True) -> None:
    """
    Plot log1p histograms for 'TotalPremium' and 'TotalClaims' showing distributions
    after log-transform (standard approach for heavy-tailed insurance variables).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing 'TotalPremium' and 'TotalClaims'.
    save : bool, optional
        If True, save the plot to `reports/figures/log_transforms.png`.

    Returns
    -------
    None

    Example
    -------
    >>> plot_log_transforms(df)
    """
    cols = ['TotalPremium', 'TotalClaims']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    fig.suptitle('Log-Transformed Financial Variables', fontsize=18, fontweight='bold', y=1.02)

    for ax, col in zip(axes, cols):
        data_positive = df[df[col] > 0][col]
        sns.histplot(np.log1p(data_positive), bins=80, kde=True,
                     color='#1f77b4' if col == 'TotalPremium' else '#d62728',
                     alpha=0.8, edgecolor='black', linewidth=0.5, ax=ax)
        ax.set_title(f'log1p({col})', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        _save_fig("log_transforms.png")
    plt.show()
