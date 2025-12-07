"""
Generate a comprehensive test dataset for AutoML system
This dataset includes various data quality issues to test all features
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Create datasets directory if it doesn't exist
Path("datasets").mkdir(exist_ok=True)

print("Generating comprehensive test dataset...")

# Dataset size
n_samples = 2000

print(f"Creating {n_samples} samples with multiple features and issues...")

# ============================================
# CUSTOMER CHURN PREDICTION DATASET
# ============================================

data = {}

# 1. CUSTOMER ID (will be constant/near-constant after processing)
data['CustomerID'] = range(1000, 1000 + n_samples)

# 2. DEMOGRAPHIC FEATURES
print("   - Adding demographic features...")

# Age (with some outliers and missing values)
ages = np.random.normal(45, 15, n_samples).clip(18, 90)
# Add some extreme outliers
outlier_indices = np.random.choice(n_samples, 50, replace=False)
ages[outlier_indices] = np.random.choice([15, 120, 150], 50)
# Add missing values
missing_age_indices = np.random.choice(n_samples, 150, replace=False)
ages[missing_age_indices] = np.nan
data['Age'] = ages

# Gender (slightly imbalanced)
data['Gender'] = np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04])

# Location - High Cardinality Feature
cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
          'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
          'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte',
          'Seattle', 'Denver', 'Boston', 'Portland', 'Las Vegas',
          'Detroit', 'Memphis', 'Nashville', 'Baltimore', 'Louisville'] + \
         [f'City_{i}' for i in range(25)]  # Add more rare cities
city_probs = [0.1, 0.08, 0.07, 0.06, 0.05] + [0.64 / 45] * 45
data['City'] = np.random.choice(cities, n_samples, p=city_probs)

# 3. ACCOUNT FEATURES
print("   - Adding account features...")

# Tenure (months) - with some missing values
tenure = np.random.gamma(3, 8, n_samples).clip(0, 72)
missing_tenure_indices = np.random.choice(n_samples, 80, replace=False)
tenure[missing_tenure_indices] = np.nan
data['Tenure_Months'] = tenure

# Monthly Charges (with outliers)
monthly_charges = np.random.gamma(4, 20, n_samples)
# Add outliers
outlier_charges = np.random.choice(n_samples, 60, replace=False)
monthly_charges[outlier_charges] = np.random.uniform(500, 1000, 60)
data['Monthly_Charges'] = monthly_charges

# Total Charges (derived from tenure * monthly, with some noise)
total_charges = np.where(
    np.isnan(tenure) | np.isnan(monthly_charges),
    np.nan,
    tenure * monthly_charges * np.random.uniform(0.8, 1.2, n_samples)
)
data['Total_Charges'] = total_charges

# Contract Type
data['Contract_Type'] = np.random.choice(
    ['Month-to-Month', 'One Year', 'Two Year'], 
    n_samples, 
    p=[0.5, 0.3, 0.2]
)

# Payment Method - with missing values
payment_methods = np.random.choice(
    ['Credit Card', 'Bank Transfer', 'Electronic Check', 'Mailed Check', None],
    n_samples,
    p=[0.30, 0.25, 0.20, 0.15, 0.10]
)
data['Payment_Method'] = payment_methods

# 4. SERVICE FEATURES
print("   - Adding service features...")

# Internet Service
data['Internet_Service'] = np.random.choice(
    ['Fiber Optic', 'DSL', 'No'], 
    n_samples, 
    p=[0.45, 0.35, 0.20]
)

# Online Security
data['Online_Security'] = np.random.choice(['Yes', 'No', 'No internet service'], n_samples)

# Tech Support
data['Tech_Support'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])

# Streaming TV
data['Streaming_TV'] = np.random.choice(['Yes', 'No'], n_samples)

# Streaming Movies
data['Streaming_Movies'] = np.random.choice(['Yes', 'No'], n_samples)

# 5. BEHAVIORAL FEATURES
print("   - Adding behavioral features...")

# Number of Support Tickets (with outliers)
support_tickets = np.random.poisson(2, n_samples)
outlier_tickets = np.random.choice(n_samples, 40, replace=False)
support_tickets[outlier_tickets] = np.random.randint(20, 50, 40)
data['Support_Tickets'] = support_tickets

# Late Payments (with missing values)
late_payments = np.random.binomial(5, 0.2, n_samples)
missing_late = np.random.choice(n_samples, 100, replace=False)
late_payments = late_payments.astype(float)
late_payments[missing_late] = np.nan
data['Late_Payments'] = late_payments

# Contract Changes
data['Contract_Changes'] = np.random.poisson(1, n_samples)

# Customer Satisfaction Score (1-5, with some missing)
satisfaction = np.random.choice([1, 2, 3, 4, 5, np.nan], n_samples, p=[0.05, 0.10, 0.25, 0.35, 0.20, 0.05])
data['Satisfaction_Score'] = satisfaction

# 6. CONSTANT/NEAR-CONSTANT FEATURES (to test removal)
print("   - Adding constant features (should be detected)...")

# Almost constant feature
data['Account_Status'] = ['Active'] * (n_samples - 5) + ['Inactive'] * 5

# Completely constant feature
data['Company_Name'] = ['TelecomCorp'] * n_samples

# 7. DUPLICATE ROWS
print("   - Creating some duplicate rows...")

# Create DataFrame
df = pd.DataFrame(data)

# Add some duplicate rows (about 3%)
duplicate_indices = np.random.choice(n_samples, 60, replace=False)
duplicates = df.iloc[duplicate_indices].copy()
df = pd.concat([df, duplicates], ignore_index=True)

# 8. TARGET VARIABLE - CHURN (IMBALANCED)
print("   - Generating imbalanced target variable (Churn)...")

# Create churn based on features (more realistic)
churn_probability = (
    (df['Contract_Type'] == 'Month-to-Month').astype(int) * 0.3 +
    (df['Support_Tickets'] > 5).astype(int) * 0.2 +
    (df['Tenure_Months'].fillna(0) < 6).astype(int) * 0.25 +
    (df['Late_Payments'].fillna(0) > 2).astype(int) * 0.15 +
    (df['Satisfaction_Score'].fillna(3) < 3).astype(int) * 0.2 +
    np.random.random(len(df)) * 0.3  # Random component
)

# Make it imbalanced (70% No Churn, 30% Churn)
churn_probability = churn_probability * 0.5  # Scale down to make it imbalanced
df['Churn'] = (churn_probability > 0.35).astype(int)

# Further imbalance - change some 1s to 0s
churn_indices = df[df['Churn'] == 1].index
change_indices = np.random.choice(churn_indices, int(len(churn_indices) * 0.4), replace=False)
df.loc[change_indices, 'Churn'] = 0

# 9. ADD SOME MORE MISSING VALUES RANDOMLY
print("   - Adding additional random missing values...")

# Randomly add missing values to numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['CustomerID', 'Churn'])
for col in numeric_cols:
    if np.random.random() > 0.7:  # 30% chance to add missing values
        missing_indices = np.random.choice(len(df), int(len(df) * 0.05), replace=False)
        df.loc[missing_indices, col] = np.nan

# Save dataset
output_path = 'datasets/customer_churn_test.csv'
df.to_csv(output_path, index=False)

print(f"\nDataset saved to: {output_path}")
print(f"\nDataset Statistics:")
print(f"   - Total Rows: {len(df):,}")
print(f"   - Total Columns: {len(df.columns)}")
print(f"   - Target Column: Churn")
print(f"\nClass Distribution:")
print(f"   - No Churn (0): {(df['Churn'] == 0).sum():,} ({(df['Churn'] == 0).sum() / len(df) * 100:.1f}%)")
print(f"   - Churn (1): {(df['Churn'] == 1).sum():,} ({(df['Churn'] == 1).sum() / len(df) * 100:.1f}%)")
print(f"\nIssues Intentionally Included:")
print(f"   - Missing Values: {df.isnull().sum().sum():,} cells")
print(f"   - Outliers: Present in Age, Monthly_Charges, Support_Tickets")
print(f"   - Class Imbalance: Churn is imbalanced ({(df['Churn'] == 1).sum() / len(df) * 100:.1f}% minority class)")
print(f"   - High Cardinality: City column has 50 unique values")
print(f"   - Constant Features: Account_Status, Company_Name")
print(f"   - Duplicate Rows: ~60 duplicate records")
print(f"\nFeature Types:")
print(f"   - Numerical: {len(df.select_dtypes(include=[np.number]).columns) - 1} (excluding Churn)")
print(f"   - Categorical: {len(df.select_dtypes(include=['object']).columns)}")

print("\n" + "="*60)
print("✅ Test dataset generation complete!")
print("="*60)
print("\nYou can now:")
print("   1. Upload 'datasets/customer_churn_test.csv' in the app")
print("   2. Select 'Churn' as the target column")
print("   3. Test all EDA, issue detection, and preprocessing features!")
print("\nThis dataset will trigger:")
print("   ✓ Missing value detection and handling")
print("   ✓ Outlier detection and treatment")
print("   ✓ Class imbalance warning and SMOTE option")
print("   ✓ High cardinality feature detection")
print("   ✓ Constant feature removal")
print("   ✓ Duplicate row detection")
print("   ✓ All preprocessing and model training features")
