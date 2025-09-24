import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import xgboost as xgb
import joblib

class InsuranceMLModels:
    """Machine learning models for insurance analytics."""
    
    def __init__(self, df):
        self.df = df
        self.models = {}
        self.scalers = {}
        self.encoders = {}
    
    def prepare_features(self, target_col, exclude_cols=None):
        """Prepare features for modeling."""
        if exclude_cols is None:
            exclude_cols = ['UnderwrittenCoverID', 'PolicyID', 'TransactionMonth']
        
        # Select features
        feature_cols = [col for col in self.df.columns 
                       if col not in exclude_cols + [target_col]]
        
        X = self.df[feature_cols].copy()
        y = self.df[target_col].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col].fillna('Unknown'))
            else:
                X[col] = self.encoders[col].transform(X[col].fillna('Unknown'))
        
        # Handle missing values
        X = X.fillna(X.median())
        
        return X, y
    
    def train_claim_severity_model(self):
        """Train model to predict claim severity (total claims amount)."""
        print("Training Claim Severity Model...")
        
        # Filter to policies with claims
        claims_data = self.df[self.df['HasClaim'] == 1].copy()
        
        if len(claims_data) == 0:
            print("No claims data available for training")
            return None
        
        X, y = self.prepare_features('TotalClaims', 
                                   exclude_cols=['HasClaim', 'Margin', 'ClaimsRatio'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            if name == 'linear_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"{name}: RMSE={rmse:.2f}, R²={r2:.3f}")
        
        # Select best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
        self.models['claim_severity'] = results[best_model_name]['model']
        self.scalers['claim_severity'] = scaler if best_model_name == 'linear_regression' else None
        
        print(f"Best model: {best_model_name}")
        return results
    
    def train_premium_optimization_model(self):
        """Train model for premium optimization."""
        print("\nTraining Premium Optimization Model...")
        
        X, y = self.prepare_features('TotalPremium', 
                                   exclude_cols=['Margin', 'PremiumRatio'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            if name == 'linear_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"{name}: RMSE={rmse:.2f}, R²={r2:.3f}")
        
        # Select best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
        self.models['premium_optimization'] = results[best_model_name]['model']
        self.scalers['premium_optimization'] = scaler if best_model_name == 'linear_regression' else None
        
        print(f"Best model: {best_model_name}")
        return results
    
    def train_risk_classification_model(self):
        """Train binary classification model for claim probability."""
        print("\nTraining Risk Classification Model...")
        
        X, y = self.prepare_features('HasClaim', 
                                   exclude_cols=['TotalClaims', 'Margin', 'ClaimsRatio'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            if name == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy
            }
            
            print(f"{name}: Accuracy={accuracy:.3f}")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        self.models['risk_classification'] = results[best_model_name]['model']
        self.scalers['risk_classification'] = scaler if best_model_name == 'logistic_regression' else None
        
        print(f"Best model: {best_model_name}")
        return results
    
    def save_models(self, model_dir='models'):
        """Save trained models."""
        for model_name, model in self.models.items():
            joblib.dump(model, f'{model_dir}/{model_name}_model.pkl')
            if self.scalers.get(model_name):
                joblib.dump(self.scalers[model_name], f'{model_dir}/{model_name}_scaler.pkl')
        
        # Save encoders
        joblib.dump(self.encoders, f'{model_dir}/encoders.pkl')
        print(f"Models saved to {model_dir}/")
    
    def train_all_models(self):
        """Train all models."""
        results = {}
        results['claim_severity'] = self.train_claim_severity_model()
        results['premium_optimization'] = self.train_premium_optimization_model()
        results['risk_classification'] = self.train_risk_classification_model()
        return results