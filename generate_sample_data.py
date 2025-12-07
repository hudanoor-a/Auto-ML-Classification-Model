"""
Generate comprehensive sample dataset for AutoML testing
Creates a realistic customer churn dataset with various features and data quality issues
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

print("Generating comprehensive sample dataset...")

# Dataset size - make it larger for better training
n_samples = 1000

print(f"Creating {n_samples} samples with realistic features...")

# ============================================
# CUSTOMER CHURN PREDICTION DATASET
# ============================================

data = {}

# 1. CUSTOMER ID
data['CustomerID'] = range(10000, 10000 + n_samples)

# 2. DEMOGRAPHIC FEATURES
print("   - Adding demographic features...")

# Age (with some outliers and missing values)
ages = np.random.normal(42, 13, n_samples).clip(18, 80)
# Add some outliers
outlier_indices = np.random.choice(n_samples, 25, replace=False)
ages[outlier_indices] = np.random.choice([15, 95, 100], 25)
# Add missing  values
missing_age_indices = np.random.choice(n_samples, 50, replace=False)
ages[missing_age_indices] = np.nan
data['Age'] = ages

# Gender
data['Gender'] = np.random.choice(['Male', 'Female'], n_samples, p=[0.52, 0.48])

# Location
cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
          'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
data['City'] = np.random.choice(cities, n_samples, p=[0.15, 0.12, 0.11, 0.10, 0.09,
                                                        0.09, 0.09, 0.09, 0.08, 0.08])

# 3. ACCOUNT FEATURES
print("   - Adding account features...")

# Tenure (months)
tenure = np.random.gamma(3, 6, n_samples).clip(1, 72)
missing_tenure_indices = np.random.choice(n_samples, 30, replace=False)
tenure[missing_tenure_indices] = np.nan
data['Tenure_Months'] = tenure

# Monthly Charges (with outliers)
monthly_charges = np.random.gamma(4, 15, n_samples).clip(10, 200)
# Add outliers
outlier_charges = np.random.choice(n_samples, 30, replace=False)
monthly_charges[outlier_charges] = np.random.uniform(300, 500, 30)
data['Monthly_Charges'] = monthly_charges

# Total Charges
total_charges = np.where(
    np.isnan(tenure),
    np.nan,
    tenure * monthly_charges * np.random.uniform(0.9, 1.1, n_samples)
)
data['Total_Charges'] = total_charges

# Contract Type
data['Contract_Type'] = np.random.choice(
    ['Month-to-Month', 'One Year', 'Two Year'], 
    n_samples, 
    p=[0.55, 0.25, 0.20]
)

# Payment Method
payment_methods = np.random.choice(
    ['Credit Card', 'Bank Transfer', 'Electronic Check', 'Mailed Check'],
    n_samples,
    p=[0.30, 0.25, 0.25, 0.20]
)
# Add some missing values
missing_payment_indices = np.random.choice(n_samples, 20, replace=False)
payment_methods = payment_methods.astype(object)
payment_methods[missing_payment_indices] = None
data['Payment_Method'] = payment_methods

# 4. SERVICE FEATURES
print("   - Adding service features...")

# Internet Service
data['Internet_Service'] = np.random.choice(
    ['Fiber Optic', 'DSL', 'No'], 
    n_samples, 
    p=[0.50, 0.35, 0.15]
)

# Phone Service
data['Phone_Service'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10])

# Online Security
data['Online_Security'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65])

# Tech Support
data['Tech_Support'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.32, 0.68])

# Streaming TV
data['Streaming_TV'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.45, 0.55])

# 5. BEHAVIORAL FEATURES
print("   - Adding behavioral features...")

# Number of Support Tickets (with outliers)
support_tickets = np.random.poisson(1.5, n_samples)
outlier_tickets = np.random.choice(n_samples, 20, replace=False)
support_tickets[outlier_tickets] = np.random.randint(15, 30, 20)
data['Support_Tickets'] = support_tickets

# Late Payments
late_payments = np.random.binomial(4, 0.15, n_samples)
missing_late = np.random.choice(n_samples, 25, replace=False)
late_payments = late_payments.astype(float)
late_payments[missing_late] = np.nan
data['Late_Payments'] = late_payments

# Contract Changes
data['Contract_Changes'] = np.random.poisson(0.8, n_samples)

# Customer Satisfaction Score (1-5)
satisfaction = np.random.choice([1, 2, 3, 4, 5], n_samples, 
                               p=[0.05, 0.10, 0.30, 0.35, 0.20])
# Add some missing
missing_satisfaction = np.random.choice(n_samples, 15, replace=False)
satisfaction = satisfaction.astype(float)
satisfaction[missing_satisfaction] = np.nan
data['Satisfaction_Score'] = satisfaction

# Create DataFrame
df = pd.DataFrame(data)

# 6. ADD SOME DUPLICATE ROWS (about 2%)
print("   - Creating duplicate rows...")
duplicate_indices = np.random.choice(n_samples, 20, replace=False)
duplicates = df.iloc[duplicate_indices].copy()
df = pd.concat([df, duplicates], ignore_index=True)

# 7. TARGET VARIABLE - CHURN (BALANCED for better model training)
print("   - Generating balanced target variable...")

# Create churn based on realistic features
churn_probability = (
    (df['Contract_Type'] == 'Month-to-Month').astype(int) * 0.25 +
    (df['Support_Tickets'] > 5).astype(int) * 0.20 +
    (df['Tenure_Months'].fillna(0) < 12).astype(int) * 0.20 +
    (df['Late_Payments'].fillna(0) > 2).astype(int) * 0.15 +
    (df['Satisfaction_Score'].fillna(3) < 3).astype(int) * 0.20 +
    (df['Internet_Service'] == 'Fiber Optic').astype(int) * 0.10 +
    np.random.random(len(df)) * 0.3  # Random component
)

# Create more balanced classes (aim for 40-60% split)
churn_probability = churn_probability * 0.7  # Scale to get better balance
df['Churn'] = (churn_probability > 0.42).astype(int)

# Fine-tune for balance
churn_rate = df['Churn'].mean()
if churn_rate < 0.35:  # If too few churns
    # Flip some 0s to 1s
    no_churn_indices = df[df['Churn'] == 0].index
    flip_count = int(len(df) * 0.05)
    flip_indices = np.random.choice(no_churn_indices, flip_count, replace=False)
    df.loc[flip_indices, 'Churn'] = 1
elif churn_rate > 0.65:  # If too many churns
    # Flip some 1s to 0s
    churn_indices = df[df['Churn'] == 1].index
    flip_count = int(len(df) * 0.05)
    flip_indices = np.random.choice(churn_indices, flip_count, replace=False)
    df.loc[flip_indices, 'Churn'] = 0

# Save dataset
output_path = 'data/customer_churn.csv'
df.to_csv(output_path, index=False)

print(f"\nDataset saved to: {output_path}")
print(f"\nDataset Statistics:")
print(f"   - Total Rows: {len(df):,}")
print(f"   - Total Columns: {len(df.columns)}")
print(f"   - Target Column: Churn")
print(f"\nClass Distribution:")
print(f"   - No Churn (0): {(df['Churn'] == 0).sum():,} ({(df['Churn'] == 0).sum() / len(df) * 100:.1f}%)")
print(f"   - Churn (1): {(df['Churn'] == 1).sum():,} ({(df['Churn'] == 1).sum() / len(df) * 100:.1f}%)")
print(f"\nData Quality Issues:")
print(f"   - Missing Values: {df.isnull().sum().sum():,} cells")
print(f"   - Features with Missing Data:")
for col in df.columns:
    missing = df[col].isnull().sum()
    if missing > 0:
        print(f"     • {col}: {missing} ({missing/len(df)*100:.1f}%)")
print(f"   - Outliers: Present in Age, Monthly_Charges, Support_Tickets")
print(f"   - Duplicate Rows: ~20 duplicate records")
print(f"\nFeature Types:")
print(f"   - Numerical: {len(df.select_dtypes(include=[np.number]).columns) - 1}")
print(f"   - Categorical: {len(df.select_dtypes(include=['object']).columns)}")

print("\n" + "="*60)
print("✅ Sample dataset generation complete!")
print("="*60)
print("\nUpload 'data/customer_churn.csv' in the AutoML app")
print("   and train all models successfully!")
