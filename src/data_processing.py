import pandas as pd
import numpy as np
from datetime import datetime

def load_data(file_path):
    """Load insurance data from file."""
    df = pd.read_csv(file_path, delimiter='|')
    return df

def clean_data(df):
    """Clean and preprocess the insurance data."""
    df = df.copy()
    
    # Convert date columns
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
    
    # Convert numeric columns
    numeric_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create derived features
    df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    df['VehicleAge'] = 2015 - df['RegistrationYear']
    
    # Clean categorical variables
    df['Gender'] = df['Gender'].replace('Not specified', np.nan)
    df['MaritalStatus'] = df['MaritalStatus'].replace('Not specified', np.nan)
    
    return df

def create_features(df):
    """Create additional features for modeling."""
    df = df.copy()
    
    # Premium per sum insured ratio
    df['PremiumRatio'] = df['TotalPremium'] / (df['SumInsured'] + 1)
    
    # Claims ratio
    df['ClaimsRatio'] = df['TotalClaims'] / (df['TotalPremium'] + 1)
    
    # Vehicle value categories
    df['VehicleValueCategory'] = pd.cut(df['SumInsured'], 
                                       bins=[0, 50000, 150000, 300000, np.inf],
                                       labels=['Low', 'Medium', 'High', 'Premium'])
    
    return df