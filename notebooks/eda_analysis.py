"""
Exploratory Data Analysis (EDA) Script
=======================================
Comprehensive EDA for the Bank Marketing Dataset.
Generates visualizations and business insights.

Run: python notebooks/eda_analysis.py
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Configure plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def problem_understanding():
    """Display problem context and business understanding."""
    print_section("1. PROBLEM UNDERSTANDING")

    print("""
    BUSINESS PROBLEM:
    -----------------
    A Portuguese banking institution conducts direct marketing campaigns
    (phone calls) to promote term deposit subscriptions. The bank wants
    to predict which clients are most likely to subscribe, so they can
    target campaigns more effectively and reduce costs.

    OBJECTIVE:
    ----------
    Build a predictive model to determine whether a client will subscribe
    to a term deposit (binary classification: yes/no).

    PROBLEM STATEMENT:
    ------------------
    Given client demographic data, financial information, and campaign
    interaction history, predict the probability of term deposit subscription.

    SUCCESS METRICS:
    ----------------
    - Accuracy:   Overall correctness of predictions
    - Precision:  Of predicted subscribers, how many actually subscribe?
                  (Reduces wasted marketing spend)
    - Recall:     Of actual subscribers, how many did we identify?
                  (Captures potential revenue)
    - F1-Score:   Harmonic mean of Precision & Recall (PRIMARY METRIC)
    - ROC-AUC:    Model's ability to distinguish classes across thresholds

    WHY F1-SCORE?
    Since the dataset is imbalanced (~88% No, ~12% Yes), accuracy alone
    is misleading. F1-Score balances precision and recall, making it the
    best single metric for this problem.
    """)


def data_understanding(df):
    """Display data overview and column descriptions."""
    print_section("2. DATA UNDERSTANDING")

    print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    print("Column Information:")
    print("-" * 60)
    for col in df.columns:
        dtype = df[col].dtype
        nunique = df[col].nunique()
        missing = df[col].isnull().sum()
        print(f"  {col:<15} | Type: {str(dtype):<10} | Unique: {nunique:<6} | Missing: {missing}")

    print(f"\nMissing Values Total: {df.isnull().sum().sum()}")

    print("""
    KEY COLUMNS EXPLAINED:
    ----------------------
    - pdays:     Number of days since the client was last contacted from a
                 PREVIOUS campaign. Value -1 means the client was NOT previously
                 contacted. Higher values = longer since last contact.

    - previous:  Number of contacts performed BEFORE this campaign for this
                 client. 0 means no prior contact history.

    - poutcome:  Outcome of the PREVIOUS marketing campaign.
                 Values: 'success', 'failure', 'other', 'unknown'
                 'success' from previous campaign strongly predicts current subscription.

    - y:         TARGET VARIABLE - Has the client subscribed to a term deposit?
                 Binary: 'yes' or 'no'
    """)

    print("\nTarget Variable Distribution:")
    print(df['y'].value_counts().to_string())
    print(f"\nClass Imbalance Ratio: {df['y'].value_counts()['no'] / df['y'].value_counts()['yes']:.1f}:1 (No:Yes)")


def univariate_analysis(df, output_dir):
    """Perform univariate analysis with visualizations."""
    print_section("3a. UNIVARIATE ANALYSIS")

    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

    # Numeric distributions
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        df[col].hist(bins=40, ax=ax, color='#6366F1', edgecolor='white', alpha=0.8)
        ax.set_title(f'Distribution of {col}', fontweight='bold', fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    # Hide unused subplots
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Numeric Feature Distributions', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'univariate_numeric.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: univariate_numeric.png")

    # Categorical distributions
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()
    for i, col in enumerate(categorical_cols):
        ax = axes[i]
        df[col].value_counts().plot(kind='bar', ax=ax, color='#8B5CF6', edgecolor='white', alpha=0.8)
        ax.set_title(f'Distribution of {col}', fontweight='bold', fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', alpha=0.3)
    plt.suptitle('Categorical Feature Distributions', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'univariate_categorical.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: univariate_categorical.png")

    # Target distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#EF4444', '#10B981']
    df['y'].value_counts().plot(kind='bar', ax=ax, color=colors, edgecolor='white')
    ax.set_title('Target Variable Distribution (y)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Subscription')
    ax.set_ylabel('Count')
    for i, v in enumerate(df['y'].value_counts().values):
        ax.text(i, v + 200, f'{v:,} ({v/len(df):.1%})', ha='center', fontweight='bold', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: target_distribution.png")


def bivariate_analysis(df, output_dir):
    """Perform bivariate analysis — features vs target."""
    print_section("3b. BIVARIATE ANALYSIS")

    df_encoded = df.copy()
    df_encoded['y_num'] = df_encoded['y'].map({'no': 0, 'yes': 1})

    # Subscription rate by job
    fig, ax = plt.subplots(figsize=(14, 6))
    sub_rate = df_encoded.groupby('job')['y_num'].mean().sort_values(ascending=False)
    bars = sub_rate.plot(kind='bar', ax=ax, color='#6366F1', edgecolor='white', alpha=0.85)
    ax.set_title('Subscription Rate by Job Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Job')
    ax.set_ylabel('Subscription Rate')
    ax.axhline(y=df_encoded['y_num'].mean(), color='#EF4444', linestyle='--', linewidth=2, label=f'Overall Rate ({df_encoded["y_num"].mean():.2%})')
    ax.legend()
    for i, v in enumerate(sub_rate.values):
        ax.text(i, v + 0.005, f'{v:.1%}', ha='center', fontsize=9)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sub_rate_by_job.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: sub_rate_by_job.png")

    # Subscription rate by education
    fig, ax = plt.subplots(figsize=(10, 5))
    sub_rate = df_encoded.groupby('education')['y_num'].mean().sort_values(ascending=False)
    sub_rate.plot(kind='bar', ax=ax, color='#8B5CF6', edgecolor='white', alpha=0.85)
    ax.set_title('Subscription Rate by Education Level', fontsize=14, fontweight='bold')
    ax.set_xlabel('Education')
    ax.set_ylabel('Subscription Rate')
    ax.axhline(y=df_encoded['y_num'].mean(), color='#EF4444', linestyle='--', linewidth=2, label='Overall Rate')
    ax.legend()
    for i, v in enumerate(sub_rate.values):
        ax.text(i, v + 0.005, f'{v:.1%}', ha='center', fontsize=10)
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sub_rate_by_education.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: sub_rate_by_education.png")

    # Subscription rate by previous outcome (KEY FEATURE)
    fig, ax = plt.subplots(figsize=(10, 5))
    sub_rate = df_encoded.groupby('poutcome')['y_num'].mean().sort_values(ascending=False)
    colors_po = ['#10B981' if v > 0.3 else '#6366F1' for v in sub_rate.values]
    sub_rate.plot(kind='bar', ax=ax, color=colors_po, edgecolor='white', alpha=0.85)
    ax.set_title('Subscription Rate by Previous Campaign Outcome', fontsize=14, fontweight='bold')
    ax.set_xlabel('Previous Outcome')
    ax.set_ylabel('Subscription Rate')
    for i, v in enumerate(sub_rate.values):
        ax.text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sub_rate_by_poutcome.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: sub_rate_by_poutcome.png")

    # Duration vs Subscription
    fig, ax = plt.subplots(figsize=(12, 5))
    df_encoded.boxplot(column='duration', by='y', ax=ax,
                       boxprops=dict(color='#6366F1'), medianprops=dict(color='#EF4444', linewidth=2))
    ax.set_title('Call Duration by Subscription Status', fontsize=14, fontweight='bold')
    ax.set_xlabel('Subscription (y)')
    ax.set_ylabel('Duration (seconds)')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_by_target.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: duration_by_target.png")

    # Age distribution by subscription
    fig, ax = plt.subplots(figsize=(12, 5))
    df_encoded[df_encoded['y'] == 'no']['age'].hist(bins=40, alpha=0.6, label='No', color='#EF4444', ax=ax)
    df_encoded[df_encoded['y'] == 'yes']['age'].hist(bins=40, alpha=0.6, label='Yes', color='#10B981', ax=ax)
    ax.set_title('Age Distribution by Subscription Status', fontsize=14, fontweight='bold')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_by_target.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: age_by_target.png")

    # Monthly campaign volume and success
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month_counts = df_encoded.groupby('month').size().reindex(month_order)
    month_rate = df_encoded.groupby('month')['y_num'].mean().reindex(month_order)

    month_counts.plot(kind='bar', ax=axes[0], color='#6366F1', edgecolor='white', alpha=0.8)
    axes[0].set_title('Campaign Volume by Month', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Number of Contacts')
    axes[0].tick_params(axis='x', rotation=45)

    month_rate.plot(kind='bar', ax=axes[1], color='#10B981', edgecolor='white', alpha=0.8)
    axes[1].set_title('Subscription Rate by Month', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Subscription Rate')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(y=df_encoded['y_num'].mean(), color='#EF4444', linestyle='--', linewidth=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: monthly_analysis.png")


def correlation_analysis(df, output_dir):
    """Perform correlation analysis on numeric features."""
    print_section("3c. CORRELATION ANALYSIS")

    df_corr = df.copy()
    df_corr['y_num'] = df_corr['y'].map({'no': 0, 'yes': 1})

    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', 'y_num']
    corr_matrix = df_corr[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Matrix (Numeric Features + Target)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: correlation_matrix.png")

    # Correlation with target
    print("\n  Correlation with Target (y):")
    target_corr = corr_matrix['y_num'].drop('y_num').sort_values(ascending=False)
    for feat, val in target_corr.items():
        bar = '+' * int(abs(val) * 30)
        sign = '+' if val > 0 else '-'
        print(f"    {feat:<12}: {val:+.4f}  {sign}{bar}")


def business_insights(df):
    """Extract and display key business insights."""
    print_section("4. KEY BUSINESS INSIGHTS")

    df_copy = df.copy()
    df_copy['y_num'] = df_copy['y'].map({'no': 0, 'yes': 1})

    overall_rate = df_copy['y_num'].mean()

    print(f"""
    INSIGHT 1: Previous Campaign Outcome is the STRONGEST Predictor
    ---------------------------------------------------------------
    Clients with 'success' in poutcome have a {df_copy[df_copy['poutcome']=='success']['y_num'].mean():.1%} subscription rate
    vs {overall_rate:.1%} overall. This is {df_copy[df_copy['poutcome']=='success']['y_num'].mean()/overall_rate:.1f}x the average!
    >> ACTION: Prioritize re-contacting previously successful clients.

    INSIGHT 2: Call Duration is Highly Correlated with Subscription
    ---------------------------------------------------------------
    Average duration for subscribers: {df_copy[df_copy['y']=='yes']['duration'].mean():.0f} seconds
    Average duration for non-subscribers: {df_copy[df_copy['y']=='no']['duration'].mean():.0f} seconds
    >> CAVEAT: Duration is only known AFTER the call. Cannot use for targeting.
    >> ACTION: Train agents to keep clients engaged longer on calls.

    INSIGHT 3: Students and Retired Clients Subscribe Most
    ------------------------------------------------------
    Students: {df_copy[df_copy['job']=='student']['y_num'].mean():.1%} subscription rate
    Retired:  {df_copy[df_copy['job']=='retired']['y_num'].mean():.1%} subscription rate
    Blue-collar: {df_copy[df_copy['job']=='blue-collar']['y_num'].mean():.1%} subscription rate
    >> ACTION: Focus campaigns on students and retired individuals.

    INSIGHT 4: Severe Class Imbalance (88:12)
    ------------------------------------------
    Only {overall_rate:.1%} of clients subscribed. This means:
    - A naive "always predict No" model would be {1-overall_rate:.1%} accurate!
    - We need SMOTE + threshold tuning to handle this properly.
    >> ACTION: Use F1-Score as primary metric, not accuracy.

    INSIGHT 5: March, September, October Have Highest Success Rates
    ---------------------------------------------------------------
    These months show above-average subscription rates but lower volume.
    May has the highest volume but a below-average success rate.
    >> ACTION: Consider shifting campaign timing to high-success months.

    INSIGHT 6: Clients Without Prior Contact Dominate the Dataset
    -------------------------------------------------------------
    {(df_copy['pdays']==-1).mean():.1%} of clients were never previously contacted (pdays=-1).
    {(df_copy['previous']==0).mean():.1%} have zero previous contacts.
    >> ACTION: Build separate strategies for new vs. returning clients.

    INSIGHT 7: Housing Loan Holders Subscribe Less
    ------------------------------------------------
    With housing loan: {df_copy[df_copy['housing']=='yes']['y_num'].mean():.1%} subscription rate
    Without housing loan: {df_copy[df_copy['housing']=='no']['y_num'].mean():.1%} subscription rate
    >> ACTION: Adjust targeting to account for existing financial commitments.
    """)


def main():
    """Run the complete EDA pipeline."""
    print("\n" + "#" * 70)
    print("#  BANK MARKETING DATASET - EXPLORATORY DATA ANALYSIS")
    print("#" * 70)

    # Load data
    data_path = os.path.join(PROJECT_ROOT, 'data', 'data.csv')
    df = pd.read_csv(data_path, sep=';')

    # Create output directory for plots
    output_dir = os.path.join(os.path.dirname(__file__), 'eda_plots')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nPlots will be saved to: {output_dir}\n")

    # Run all analyses
    problem_understanding()
    data_understanding(df)
    univariate_analysis(df, output_dir)
    bivariate_analysis(df, output_dir)
    correlation_analysis(df, output_dir)
    business_insights(df)

    print_section("EDA COMPLETE")
    print(f"  All plots saved to: {output_dir}")
    print(f"  Total plots generated: {len([f for f in os.listdir(output_dir) if f.endswith('.png')])}")


if __name__ == '__main__':
    main()
