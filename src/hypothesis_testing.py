import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

class HypothesisTests:
    """Class for conducting A/B hypothesis tests on insurance data."""
    
    def __init__(self, df):
        self.df = df
        self.results = {}
    
    def test_risk_across_provinces(self, alpha=0.05):
        """Test if there are risk differences across provinces."""
        # Group by province and calculate risk metrics
        province_stats = self.df.groupby('Province').agg({
            'TotalClaims': 'mean',
            'HasClaim': 'mean',
            'TotalPremium': 'mean'
        }).reset_index()
        
        # ANOVA test for claims across provinces
        provinces = self.df['Province'].unique()
        claims_by_province = [self.df[self.df['Province'] == p]['TotalClaims'] for p in provinces]
        
        f_stat, p_value = stats.f_oneway(*claims_by_province)
        
        result = {
            'test': 'Risk differences across provinces',
            'statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'conclusion': 'Reject null hypothesis' if p_value < alpha else 'Fail to reject null hypothesis'
        }
        
        self.results['provinces'] = result
        return result
    
    def test_risk_across_zip_codes(self, alpha=0.05):
        """Test if there are risk differences between zip codes."""
        # Sample top zip codes for testing
        top_zips = self.df['PostalCode'].value_counts().head(10).index
        zip_data = self.df[self.df['PostalCode'].isin(top_zips)]
        
        # ANOVA test for claims across zip codes
        claims_by_zip = [zip_data[zip_data['PostalCode'] == z]['TotalClaims'] for z in top_zips]
        
        f_stat, p_value = stats.f_oneway(*claims_by_zip)
        
        result = {
            'test': 'Risk differences across zip codes',
            'statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'conclusion': 'Reject null hypothesis' if p_value < alpha else 'Fail to reject null hypothesis'
        }
        
        self.results['zip_codes'] = result
        return result
    
    def test_margin_differences_zip_codes(self, alpha=0.05):
        """Test if there are significant margin differences between zip codes."""
        top_zips = self.df['PostalCode'].value_counts().head(10).index
        zip_data = self.df[self.df['PostalCode'].isin(top_zips)]
        
        # ANOVA test for margins across zip codes
        margins_by_zip = [zip_data[zip_data['PostalCode'] == z]['Margin'] for z in top_zips]
        
        f_stat, p_value = stats.f_oneway(*margins_by_zip)
        
        result = {
            'test': 'Margin differences across zip codes',
            'statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'conclusion': 'Reject null hypothesis' if p_value < alpha else 'Fail to reject null hypothesis'
        }
        
        self.results['margins_zip'] = result
        return result
    
    def test_risk_differences_gender(self, alpha=0.05):
        """Test if there are significant risk differences between genders."""
        # Filter out missing gender data
        gender_data = self.df.dropna(subset=['Gender'])
        
        if len(gender_data['Gender'].unique()) < 2:
            return {'error': 'Insufficient gender categories for testing'}
        
        # T-test for claims between genders
        male_claims = gender_data[gender_data['Gender'] == 'Male']['TotalClaims']
        female_claims = gender_data[gender_data['Gender'] == 'Female']['TotalClaims']
        
        t_stat, p_value = ttest_ind(male_claims, female_claims)
        
        result = {
            'test': 'Risk differences between genders',
            'statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'conclusion': 'Reject null hypothesis' if p_value < alpha else 'Fail to reject null hypothesis'
        }
        
        self.results['gender'] = result
        return result
    
    def run_all_tests(self, alpha=0.05):
        """Run all hypothesis tests."""
        print("Running A/B Hypothesis Tests...")
        print("=" * 50)
        
        tests = [
            self.test_risk_across_provinces,
            self.test_risk_across_zip_codes,
            self.test_margin_differences_zip_codes,
            self.test_risk_differences_gender
        ]
        
        for test in tests:
            result = test(alpha)
            if 'error' not in result:
                print(f"\n{result['test']}:")
                print(f"  Statistic: {result['statistic']:.4f}")
                print(f"  P-value: {result['p_value']:.4f}")
                print(f"  Significant: {result['significant']}")
                print(f"  Conclusion: {result['conclusion']}")
        
        return self.results