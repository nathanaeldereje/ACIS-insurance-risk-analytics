import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# -------------------------------------------------
# 1. FINANCIAL DISTRIBUTIONS (Histogram + Boxplot)
# -------------------------------------------------
def plot_financial_distributions(df):
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
    plt.show()


# -------------------------------------------------
# 2. LOSS RATIO vs CATEGORY (Province, Gender, VehicleType)
# -------------------------------------------------
def plot_loss_ratio_by(df, column, min_transactions=500, title=None, palette='Reds_r'):
    agg = (df.groupby(column, observed=False)
             .agg(TotalPremium=('TotalPremium', 'sum'),
                  TotalClaims=('TotalClaims', 'sum'),
                  Transactions=('UnderwrittenCoverID', 'count'))
             .assign(LossRatio=lambda x: x.TotalClaims / x.TotalPremium.replace(0, np.nan))
             .query(f'Transactions >= {min_transactions}')
             .sort_values('LossRatio', ascending=False)
             .reset_index())

    plt.figure(figsize=(12, 6))
    sns.barplot(data=agg,
                x='LossRatio',
                y=column,
                hue=column,
                palette=palette,
                legend=False,
                edgecolor='black')
    plt.title(title or f'Loss Ratio by {column}', fontsize=16, fontweight='bold')
    plt.xlabel('Loss Ratio')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

    return agg


# -------------------------------------------------
# 3. TEMPORAL TRENDS
# -------------------------------------------------
def plot_monthly_trends(monthly_df):
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

    plt.title('Temporal Trends: Loss Ratio & Claim Frequency', fontsize=18,
              fontweight='bold', pad=20)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Loss Ratio', color='#d62728', fontweight='bold')
    ax2.set_ylabel('Claim Frequency (%)', color='#1f77b4', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# 4. TOP RISKY MAKES
# -------------------------------------------------
def plot_top_risky_makes(makes_df, top_n=5):
    top = makes_df.sort_values('LossRatio', ascending=False).head(top_n).copy()

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(data=top, y='make', x='LossRatio', hue='make',
                     palette='Reds_r', order=top['make'],
                     legend=False, edgecolor='black', linewidth=1.5)

    for i, ratio in enumerate(top['LossRatio']):
        ax.text(ratio + 0.25, i, f'{ratio:.2f}', va='center',
                fontweight='bold', fontsize=11, color='darkred')

    ax.patches[0].set_facecolor('#8B0000')
    ax.patches[0].set_edgecolor('black')
    ax.patches[0].set_linewidth(4)

    plt.title(f'Top {top_n} Riskiest Vehicle Makes', fontsize=17, fontweight='bold')
    plt.xlabel('Loss Ratio (Claims ÷ Premium)')
    plt.ylabel('')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# 5. CORRELATION
# -------------------------------------------------
def plot_correlation(df, cols):
    plt.figure()
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Key Financial Variables')
    plt.show()


# -------------------------------------------------
# 6. LOG TRANSFORM INSIGHT
# -------------------------------------------------
def plot_log_transforms(df):
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
    plt.show()
