# End-to-End Insurance Risk Analytics & Predictive Modeling

## Overview

This project analyzes historical insurance claim data to help AlphaCare Insurance Solutions (ACIS) optimize their marketing strategy and discover "low-risk" targets for premium reduction. The analysis focuses on car insurance planning and marketing in South Africa using data from February 2014 to August 2015.

## Business Objective

- Optimize marketing strategy for car insurance
- Identify low-risk customer segments for premium reduction
- Develop predictive models for risk assessment and premium optimization
- Perform statistical analysis to validate risk hypotheses

## Project Structure

```
├── data/                   # Raw and processed datasets
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code modules
├── models/                # Trained models
├── reports/               # Analysis reports and visualizations
├── tests/                 # Unit tests
└── requirements.txt       # Project dependencies
```

## Key Features

### 1. Exploratory Data Analysis (EDA)
- Data quality assessment and missing value analysis
- Statistical distributions and outlier detection
- Correlation analysis between premiums and claims
- Geographic and temporal trend analysis

### 2. A/B Hypothesis Testing
Testing the following null hypotheses:
- No risk differences across provinces
- No risk differences between zip codes
- No significant margin differences between zip codes
- No significant risk differences between genders

### 3. Predictive Modeling
- **Claim Severity Prediction**: Predicting total claims amount for policies with claims
- **Premium Optimization**: ML model for appropriate premium calculation
- **Risk Classification**: Binary classification for claim probability

### 4. Data Version Control
- DVC implementation for reproducible data pipeline
- Version tracking for datasets and model artifacts

## Technologies Used

- **Python**: Primary programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning models
- **XGBoost**: Gradient boosting implementation
- **SHAP/LIME**: Model interpretability
- **DVC**: Data version control
- **Git**: Version control
- **GitHub Actions**: CI/CD pipeline

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd end-to-end-insurance-analytics
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize DVC:
```bash
dvc init
dvc remote add -d localstorage /path/to/your/local/storage
```

## Usage

### Data Preparation
```bash
# Add data to DVC tracking
dvc add data/raw/insurance_data.csv
dvc push
```

### Run Analysis
```bash
# Execute EDA notebook
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb

# Run hypothesis testing
python src/hypothesis_testing.py

# Train models
python src/train_models.py
```

## Data Description

The dataset contains insurance policy information with the following key categories:

- **Policy Information**: UnderwrittenCoverID, PolicyID, TransactionMonth
- **Client Demographics**: Gender, MaritalStatus, Citizenship, Language
- **Location Data**: Province, PostalCode, MainCrestaZone
- **Vehicle Details**: Make, Model, VehicleType, RegistrationYear
- **Insurance Plan**: SumInsured, CoverType, TermFrequency
- **Financial Data**: TotalPremium, TotalClaims, CalculatedPremiumPerTerm

## Key Metrics

- **Loss Ratio**: TotalClaims / TotalPremium
- **Claim Frequency**: Proportion of policies with claims
- **Claim Severity**: Average claim amount given a claim occurred
- **Margin**: TotalPremium - TotalClaims

## Model Performance

Models are evaluated using:
- **Regression**: RMSE, R-squared
- **Classification**: Accuracy, Precision, Recall, F1-score
- **Feature Importance**: SHAP values for interpretability

## Results

Key findings and business recommendations will be documented in the final report, including:
- Risk factors across different segments
- Premium optimization strategies
- Model performance comparisons
- Feature importance analysis

## Contributing

1. Create a feature branch: `git checkout -b task-<number>`
2. Make changes and commit: `git commit -m "descriptive message"`
3. Push to branch: `git push origin task-<number>`
4. Create Pull Request to main branch

## Timeline

- **Project Start**: June 11, 2025
- **Interim Submission**: June 15, 2025 (8:00 PM UTC)
- **Final Submission**: June 17, 2025 (8:00 PM UTC)

## Team

**Facilitators:**
- Mahlet
- Kerod
- Rediet
- Rehmet

## License

This project is part of the 10 Academy AI Mastery program.

## References

- [Insurance Analytics Resources](https://www.fsrao.ca/media/11501/download)
- [A/B Testing in Insurance](https://medium.com/tiket-com/a-b-testing-hypothesis-testing-f9624ea5580e)
- [DVC Documentation](https://dvc.org/doc/user-guide)
- [Statistical Modeling Guide](https://www.heavy.ai/technical-glossary/statistical-modeling)