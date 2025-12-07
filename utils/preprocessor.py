"""
Data Preprocessing Module
Applies user-approved preprocessing steps
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df, target_column, detected_issues):
    """
    Apply preprocessing based on user decisions
    
    Args:
        df: pandas DataFrame
        target_column: name of target column
        detected_issues: dictionary of detected issues
        
    Returns:
        dict: Processed data with train/test splits
    """
    st.subheader("Data Preprocessing Pipeline")
    
    if target_column is None:
        st.error("Please select a target column first!")
        return None
    
    # Make a copy of the dataframe
    df_processed = df.copy()
    
    # Display preprocessing steps
    st.info("The following preprocessing steps will be applied based on your decisions:")
    
    preprocessing_summary = []
    
    # Step 1: Remove duplicates
    if st.session_state.preprocessing_decisions.get('remove_duplicates', False):
        before_count = len(df_processed)
        df_processed = df_processed.drop_duplicates()
        after_count = len(df_processed)
        removed = before_count - after_count
        preprocessing_summary.append(f"Removed {removed} duplicate rows")
        st.success(f"Removed {removed} duplicate rows")
    
    # Step 2: Remove constant features
    if st.session_state.preprocessing_decisions.get('remove_constant_features', False):
        constant_features = [
            item['column'] for item in detected_issues.get('constant_features', [])
        ]
        if constant_features:
            df_processed = df_processed.drop(columns=constant_features)
            preprocessing_summary.append(f"Removed {len(constant_features)} constant features")
            st.success(f"Removed {len(constant_features)} constant features: {', '.join(constant_features)}")
    
    # Separate features and target
    if target_column not in df_processed.columns:
        st.error(f"Target column '{target_column}' not found in dataset!")
        return None
    
    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]
    
    # Get column types
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    st.write(f"**Features:** {len(X.columns)} ({len(numerical_cols)} numerical, {len(categorical_cols)} categorical)")
    st.write(f"**Target:** {target_column}")
    
    # Step 3: Handle missing values
    if st.session_state.preprocessing_decisions.get('fix_missing_values', False):
        st.subheader("Handling Missing Values")
        
        col1, col2 = st.columns(2)
        
        with col1:
            numerical_strategy = st.selectbox(
                "Numerical imputation strategy:",
                ["mean", "median", "most_frequent"],
                index=1
            )
        
        with col2:
            categorical_strategy = st.selectbox(
                "Categorical imputation strategy:",
                ["most_frequent", "constant"],
                index=0
            )
        
        # Impute numerical features
        if len(numerical_cols) > 0 and X[numerical_cols].isnull().sum().sum() > 0:
            num_imputer = SimpleImputer(strategy=numerical_strategy)
            X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
            preprocessing_summary.append(f"Imputed numerical features using {numerical_strategy}")
            st.success(f"Imputed numerical features using {numerical_strategy}")
        
        # Impute categorical features
        if len(categorical_cols) > 0 and X[categorical_cols].isnull().sum().sum() > 0:
            cat_imputer = SimpleImputer(strategy=categorical_strategy, fill_value='Unknown')
            X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
            preprocessing_summary.append(f"Imputed categorical features using {categorical_strategy}")
            st.success(f"Imputed categorical features using {categorical_strategy}")
    
    # Step 4: Handle outliers
    outlier_handling = st.session_state.preprocessing_decisions.get('outlier_handling', 'Keep outliers')
    
    if outlier_handling != 'Keep outliers':
        st.subheader("Handling Outliers")
        
        outlier_features = detected_issues.get('outliers', [])
        
        # Filter outlier features to only include columns that still exist in X
        valid_outlier_features = [
            info for info in outlier_features 
            if info['column'] in X.columns
        ]
        
        if outlier_handling == 'Cap outliers':
            capped_count = 0
            for outlier_info in valid_outlier_features:
                col = outlier_info['column']
                lower_bound = outlier_info['lower_bound']
                upper_bound = outlier_info['upper_bound']
                
                X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
                capped_count += 1
            
            if capped_count > 0:
                preprocessing_summary.append(f"Capped outliers in {capped_count} features")
                st.success(f"Capped outliers in {capped_count} features")
        
        elif outlier_handling == 'Remove outliers':
            if len(valid_outlier_features) > 0:
                # Create mask for outliers
                mask = pd.Series([True] * len(X), index=X.index)
                
                for outlier_info in valid_outlier_features:
                    col = outlier_info['column']
                    lower_bound = outlier_info['lower_bound']
                    upper_bound = outlier_info['upper_bound']
                    
                    mask &= (X[col] >= lower_bound) & (X[col] <= upper_bound)
                
                before_count = len(X)
                X = X[mask]
                y = y[mask]
                after_count = len(X)
                removed = before_count - after_count
                
                preprocessing_summary.append(f"Removed {removed} rows with outliers")
                st.success(f"Removed {removed} rows with outliers")
            else:
                st.info("No outlier features to process (columns may have been removed)")
    
    # Step 5: Handle high cardinality
    if st.session_state.preprocessing_decisions.get('fix_high_cardinality', False):
        st.subheader("Handling High Cardinality Features")
        
        threshold = st.session_state.preprocessing_decisions.get('cardinality_threshold', 0.05)
        high_card_features = [
            item['column'] for item in detected_issues.get('high_cardinality', [])
        ]
        
        for col in high_card_features:
            if col in X.columns:
                # Group rare categories
                value_counts = X[col].value_counts(normalize=True)
                rare_categories = value_counts[value_counts < threshold].index
                X[col] = X[col].replace(rare_categories, 'Other')
        
        preprocessing_summary.append(f"Grouped rare categories in {len(high_card_features)} features")
        st.success(f"Grouped rare categories in {len(high_card_features)} features")
    
    # Step 6: Encode categorical features
    # Re-calculate categorical columns in case some were removed
    categorical_cols_current = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # CRITICAL: Ensure NO missing values before encoding (encoding fails with NaN)
    if X.isnull().any().any():
        st.warning("Detected remaining missing values before encoding. Applying automatic imputation...")
        
        # Get current numerical and categorical columns
        num_cols_current = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Impute numerical
        if len(num_cols_current) > 0 and X[num_cols_current].isnull().any().any():
            from sklearn.impute import SimpleImputer
            num_imputer = SimpleImputer(strategy='median')
            X[num_cols_current] = num_imputer.fit_transform(X[num_cols_current])
            st.success(f"Auto-imputed {len(num_cols_current)} numerical columns")
        
        # Impute categorical
        if len(categorical_cols_current) > 0 and X[categorical_cols_current].isnull().any().any():
            from sklearn.impute import SimpleImputer
            cat_imputer = SimpleImputer(strategy='most_frequent', fill_value='Unknown')
            X[categorical_cols_current] = cat_imputer.fit_transform(X[categorical_cols_current])
            st.success(f"Auto-imputed {len(categorical_cols_current)} categorical columns")
    
    if len(categorical_cols_current) > 0:
        st.subheader("Encoding Categorical Features")
        
        encoding_method = st.radio(
            "Select encoding method:",
            ["One-Hot Encoding", "Label Encoding"],
            horizontal=True
        )
        
        if encoding_method == "One-Hot Encoding":
            # One-hot encode
            X = pd.get_dummies(X, columns=categorical_cols_current, drop_first=True)
            preprocessing_summary.append(f"Applied one-hot encoding to {len(categorical_cols_current)} features")
            st.success(f"Applied one-hot encoding to {len(categorical_cols_current)} categorical features")
        else:
            # Label encode
            for col in categorical_cols_current:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            preprocessing_summary.append(f"Applied label encoding to {len(categorical_cols_current)} features")
            st.success(f"Applied label encoding to {len(categorical_cols_current)} categorical features")
    
    # Step 7: Encode target variable (if categorical)
    target_encoder = None
    if y.dtype == 'object' or y.dtype.name == 'category':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
        st.info(f"Encoded target variable. Classes: {list(target_encoder.classes_)}")
    
    # Validation: Check if we have enough data
    if len(X) < 10:
        st.error(f"Too few samples remaining ({len(X)}). Please adjust preprocessing decisions.")
        return None
    
    # Check class distribution
    class_counts = pd.Series(y).value_counts()
    if class_counts.min() < 1:
        st.error("Some classes have no samples. Please adjust preprocessing decisions.")
        return None

    
    # Step 8: Train-test split
    st.subheader("Train-Test Split")
    
    test_size = st.session_state.get('test_size', 0.2)
    random_state = 42
    
    # Check if stratification is possible
    # Stratification requires at least 2 samples per class
    class_counts = pd.Series(y).value_counts()
    min_class_count = class_counts.min()
    can_stratify = min_class_count >= 2
    
    if not can_stratify:
        st.warning(f"Cannot use stratified split: Some classes have only {min_class_count} sample(s). Using regular random split instead.")
        stratify_param = None
    else:
        stratify_param = y
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        st.success(f"Split data: {len(X_train)} training samples, {len(X_test)} test samples")
        preprocessing_summary.append(f"Split data: {len(X_train)} train, {len(X_test)} test")
    except ValueError as e:
        # If stratification still fails, try without it
        st.warning(f"Stratified split failed: {str(e)}. Using regular random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        st.success(f"Split data: {len(X_train)} training samples, {len(X_test)} test samples")
        preprocessing_summary.append(f"Split data: {len(X_train)} train, {len(X_test)} test (no stratification)")
    
    # Step 9: Handle class imbalance (on training data only)
    imbalance_handling = st.session_state.preprocessing_decisions.get('imbalance_handling', 'No action')
    
    if imbalance_handling != 'No action':
        st.subheader("Handling Class Imbalance")
        
        if imbalance_handling == 'Use SMOTE':
            try:
                smote = SMOTE(random_state=random_state)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                preprocessing_summary.append("Applied SMOTE for class balancing")
                st.success(f"Applied SMOTE. New training size: {len(X_train)}")
            except Exception as e:
                st.warning(f"Could not apply SMOTE: {str(e)}")
        
        # Class weights will be handled during model training
    
    # Step 10: Feature scaling
    st.subheader("Feature Scaling")
    
    scaling_method = st.radio(
        "Select scaling method:",
        ["StandardScaler", "MinMaxScaler", "No Scaling"],
        horizontal=True
    )
    
    scaler = None
    if scaling_method != "No Scaling":
        if scaling_method == "StandardScaler":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        preprocessing_summary.append(f"Applied {scaling_method}")
        st.success(f"Applied {scaling_method} to features")
    
    # Display preprocessing summary
    st.markdown("---")
    st.subheader("Preprocessing Summary")
    
    for step in preprocessing_summary:
        st.write(step)
    
    # Prepare processed data
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X.columns.tolist() if isinstance(X, pd.DataFrame) else None,
        'target_encoder': target_encoder,
        'scaler': scaler,
        'preprocessing_summary': preprocessing_summary,
        'n_classes': len(np.unique(y)),
        'imbalance_handling': imbalance_handling
    }
    
    # Display final dataset info
    st.markdown("---")
    st.subheader("Processed Dataset Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Samples", len(X_train))
    
    with col2:
        st.metric("Test Samples", len(X_test))
    
    with col3:
        st.metric("Features", X_train.shape[1])
    
    with col4:
        st.metric("Classes", processed_data['n_classes'])
    
    # Show class distribution
    st.subheader("Class Distribution (Training Set)")
    
    train_class_dist = pd.Series(y_train).value_counts().sort_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(
            pd.DataFrame({
                'Class': train_class_dist.index,
                'Count': train_class_dist.values,
                'Percentage': (train_class_dist.values / len(y_train) * 100).round(2)
            }),
            use_container_width=True
        )
    
    with col2:
        st.bar_chart(train_class_dist)
    
    return processed_data
