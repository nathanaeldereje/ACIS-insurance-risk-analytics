import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_financial_distributions(df, cols=None, log_transform=False, figsize=(15,5)):
    """
    Plot histograms + KDE for selected financial columns.
    If log_transform=True, applies log1p to handle zeros and extreme skew.
    """
    if cols is None:
        cols = ['TotalPremium','TotalClaims','CalculatedPremiumPerTerm',
                'SumInsured','CustomValueEstimate','CapitalOutstanding']

    for col in cols:
        plt.figure(figsize=figsize)
        data = df[col].dropna()
        if log_transform:
            data = np.log1p(data.clip(lower=0))
            title = f"{col} (log1p scale)"
        else:
            title = col

        sns.histplot(data, bins=100, kde=True, color='skyblue')
        plt.title(f"Distribution of {title}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.show()

def plot_loss_ratio_box(df, by_column, min_transactions=500):
    grouped = df.groupby(by_column).agg(
        TotalPremium=('TotalPremium', 'sum'),
        TotalClaims=('TotalClaims', 'sum'),
        Transactions=('PolicyID','count')
    ).reset_index()
    grouped['LossRatio'] = grouped['TotalClaims'] / grouped['TotalPremium'].replace({0: np.nan})
    grouped = grouped[grouped['Transactions'] >= min_transactions]

    plt.figure(figsize=(12,6))
    sns.boxplot(x='LossRatio', y=by_column, data=grouped, color='lightgreen')
    plt.title(f"Loss Ratio Distribution by {by_column}")
    plt.xlabel("Loss Ratio")
    plt.ylabel(by_column)
    plt.show()


def plot_temporal_trends(monthly_df):
    plt.figure(figsize=(14,6))
    sns.lineplot(x='TransactionYearMonth', y='LossRatio', data=monthly_df, marker='o', label='LossRatio')
    sns.lineplot(x='TransactionYearMonth', y='ClaimFrequency', data=monthly_df, marker='x', label='ClaimFrequency')
    plt.xticks(rotation=45)
    plt.title("Monthly Loss Ratio & Claim Frequency")
    plt.xlabel("Month")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    
def plot_top_risky_makes(make_df, top_n=10):
    df_plot = make_df.sort_values('LossRatio', ascending=False).head(top_n)
    plt.figure(figsize=(12,5))
    sns.barplot(x='LossRatio', y='make', data=df_plot, palette='Reds_r')
    plt.title(f"Top {top_n} Riskiest Vehicle Makes")
    plt.xlabel("Loss Ratio")
    plt.ylabel("Make")
    plt.show()
