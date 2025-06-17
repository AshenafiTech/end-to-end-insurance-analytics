import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency


def plot_histogram(df, column, bins=50):
    df[column].hist(bins=bins)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_boxplot(df, column):
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()



def compute_claim_frequency(df):
    """
    Compute the claim frequency (proportion of policies with at least one claim).
    
    Args:
        df (pd.DataFrame): DataFrame with 'HasClaim' column (1 if claim, 0 otherwise).
        
    Returns:
        float: Claim frequency.
    """
    if 'HasClaim' not in df.columns:
        raise ValueError("DataFrame must contain 'HasClaim' column.")
    return df['HasClaim'].mean()

def compute_margin(df):
    """
    Compute the margin for each row and add as a new column 'Margin'.
    Margin = TotalPremium - TotalClaims
    
    Args:
        df (pd.DataFrame): DataFrame with 'TotalPremium' and 'TotalClaims' columns.
        
    Returns:
        pd.DataFrame: DataFrame with new 'Margin' column.
    """
    if not {'TotalPremium', 'TotalClaims'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'TotalPremium' and 'TotalClaims' columns.")
    df = df.copy()
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    return df

def segment_data(df, column, group_a, group_b):
    """
    Segment the DataFrame into two groups based on a column's values.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to segment by.
        group_a, group_b: The two values to segment on.
        
    Returns:
        (pd.DataFrame, pd.DataFrame): DataFrames for group_a and group_b.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    group_a_df = df[df[column] == group_a]
    group_b_df = df[df[column] == group_b]
    if group_a_df.empty or group_b_df.empty:
        raise ValueError("One or both groups are empty after segmentation.")
    return group_a_df, group_b_df

def run_t_test(group_a_data, group_b_data):
    """
    Run an independent t-test between two numeric data arrays.
    
    Args:
        group_a_data (array-like): Numeric data for group A.
        group_b_data (array-like): Numeric data for group B.
        
    Returns:
        tuple: (t_statistic, p_value)
    """
    if len(group_a_data) == 0 or len(group_b_data) == 0:
        raise ValueError("Input data for t-test must not be empty.")
    t_stat, p_val = ttest_ind(group_a_data, group_b_data, equal_var=False)
    return t_stat, p_val

def run_chi_squared_test(observed_values):
    """
    Run a chi-squared test on a contingency table.
    
    Args:
        observed_values (pd.DataFrame or np.ndarray): Contingency table.
        
    Returns:
        tuple: (chi2_statistic, p_value, dof, expected)
    """
    if isinstance(observed_values, pd.DataFrame):
        observed = observed_values.values
    else:
        observed = np.asarray(observed_values)
    if observed.ndim != 2:
        raise ValueError("Observed values must be a 2D array or DataFrame.")
    chi2, p, dof, expected = chi2_contingency(observed)
    return chi2, p, dof, expected