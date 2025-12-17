import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

def detect_issues(df, target_column):
    """
    Detect various data quality issues
    
    Args:
        df: pandas DataFrame
        target_column: name of the target column
        
    Returns:
        dict: Dictionary of detected issues
    """
    issues = {
        'missing_values': detect_missing_values(df),
        'outliers': detect_outliers(df, target_column),
        'class_imbalance': detect_class_imbalance(df, target_column),
        'high_cardinality': detect_high_cardinality(df, target_column),
        'constant_features': detect_constant_features(df, target_column),
        'duplicate_rows': detect_duplicate_rows(df)
    }
    
    # Note: df is passed but not strictly needed for display, 
    # included for consistency with the original function signature
    display_issues_and_get_approval(issues, df) 
    
    return issues

def detect_missing_values(df):
    """Detect missing values"""
    missing_info = []
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_pct = (missing_count / len(df)) * 100
            missing_info.append({
                'column': col,
                'missing_count': missing_count,
                'missing_percentage': missing_pct
            })
    
    return missing_info

def detect_outliers(df, target_column):
    """Detect outliers using IQR method"""
    outlier_info = []
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    
    for col in numerical_cols:
        data = df[col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        
        if outliers > 0:
            outlier_pct = (outliers / len(data)) * 100
            outlier_info.append({
                'column': col,
                'outlier_count': outliers,
                'outlier_percentage': outlier_pct,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
    
    return outlier_info

def detect_class_imbalance(df, target_column):
    """Detect class imbalance in target variable"""
    if target_column is None or target_column not in df.columns:
        return None
    
    class_counts = df[target_column].value_counts()
    total = len(df)
    
    imbalance_info = {
        'class_distribution': class_counts.to_dict(),
        'class_percentages': (class_counts / total * 100).to_dict(),
        'is_imbalanced': False,
        'imbalance_ratio': None
    }
    
    # Check if imbalanced (minority class < 20% of majority class)
    if not class_counts.empty and len(class_counts) > 1:
        max_count = class_counts.max()
        min_count = class_counts.min()
        
        if min_count / max_count < 0.2:
            imbalance_info['is_imbalanced'] = True
            imbalance_info['imbalance_ratio'] = max_count / min_count
    
    return imbalance_info

def detect_high_cardinality(df, target_column):
    """Detect high cardinality categorical features"""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    
    high_cardinality_features = []
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        unique_pct = (unique_count / len(df)) * 100
        
        # Flag if more than 50% unique or more than 20 unique categories
        if unique_pct > 50 or unique_count > 20:
            high_cardinality_features.append({
                'column': col,
                'unique_count': unique_count,
                'unique_percentage': unique_pct
            })
    
    return high_cardinality_features

def detect_constant_features(df, target_column):
    """Detect constant or near-constant features"""
    constant_features = []
    
    for col in df.columns:
        if col == target_column:
            continue
        
        # For numerical features
        if df[col].dtype in [np.int64, np.float64]:
            # Check if variance is very low
            if df[col].var() < 1e-7:
                constant_features.append({
                    'column': col,
                    'unique_values': df[col].nunique(),
                    'type': 'constant'
                })
        
        # For all features, check unique value ratio (excluding NaNs for ratio calculation simplicity)
        unique_ratio = df[col].nunique() / len(df.dropna(subset=[col])) if len(df.dropna(subset=[col])) > 0 else 0
        if unique_ratio < 0.01 and df[col].nunique() > 1: # Less than 1% unique, but has more than one value
            # Avoid duplicating features already flagged as 'constant'
            if not any(f['column'] == col and f['type'] == 'constant' for f in constant_features):
                constant_features.append({
                    'column': col,
                    'unique_values': df[col].nunique(),
                    'type': 'near-constant'
                })
    
    return constant_features

def detect_duplicate_rows(df):
    """Detect duplicate rows"""
    duplicate_count = df.duplicated().sum()
    
    return {
        'duplicate_count': duplicate_count,
        'duplicate_percentage': (duplicate_count / len(df)) * 100
    }

def display_issues_and_get_approval(issues, df):
    """Display detected issues and get user approval for fixes"""
    
    # Initialize session state for decisions if it doesn't exist
    if 'preprocessing_decisions' not in st.session_state:
        st.session_state.preprocessing_decisions = {}
        
    st.markdown("### 🔍 Detected Issues")
    
    total_issues = sum([
        len(issues['missing_values']),
        len(issues['outliers']),
        1 if issues['class_imbalance'] and issues['class_imbalance']['is_imbalanced'] else 0,
        len(issues['high_cardinality']),
        len(issues['constant_features']),
        1 if issues['duplicate_rows']['duplicate_count'] > 0 else 0
    ])
    
    if total_issues == 0:
        st.success("No major issues detected! Your dataset looks good.")
        return
    
    st.warning(f"Found {total_issues} issue(s) in your dataset")
    
    # Missing Values
    if len(issues['missing_values']) > 0:
        with st.expander("Missing Values", expanded=True):
            st.write(f"**Found missing values in {len(issues['missing_values'])} column(s)**")
            
            missing_df = pd.DataFrame(issues['missing_values'])
            st.dataframe(missing_df, use_container_width=True)
            
            st.subheader("Suggested Fix")
            st.info('''
            **Options:**
            - Impute numerical features with mean/median
            - Impute categorical features with mode
            - Remove rows with missing values (if < 5%)
            ''')
            
            # Get user approval
            is_missing_fixed = st.checkbox("I want to fix missing values", key="fix_missing")
            
            if is_missing_fixed:
                st.session_state.preprocessing_decisions['fix_missing_values'] = True
                st.success("Will handle missing values during preprocessing!")
            else:
                # FIX: Remove the key if unchecked so it disappears from the summary table
                if 'fix_missing_values' in st.session_state.preprocessing_decisions:
                    del st.session_state.preprocessing_decisions['fix_missing_values']
    
    # Outliers
    if len(issues['outliers']) > 0:
        with st.expander("Outliers", expanded=True):
            st.write(f"**Found outliers in {len(issues['outliers'])} column(s)**")
            
            outlier_df = pd.DataFrame(issues['outliers'])
            st.dataframe(outlier_df[['column', 'outlier_count', 'outlier_percentage']], 
                         use_container_width=True)
            
            st.subheader("Suggested Fix")
            st.info('''
            **Options:**
            - Cap outliers at IQR bounds (Winsorization)
            - Remove outliers (use with caution)
            - Keep outliers (they might be important)
            ''')
            
            outlier_action = st.radio(
                "How would you like to handle outliers?",
                ["Keep outliers", "Cap outliers", "Remove outliers"],
                key="outlier_action"
            )
            st.session_state.preprocessing_decisions['outlier_handling'] = outlier_action
    
    # Class Imbalance
    if issues['class_imbalance'] and issues['class_imbalance']['is_imbalanced']:
        with st.expander("Class Imbalance", expanded=True):
            imb_info = issues['class_imbalance']
            st.write("**Imbalanced class distribution detected!**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Class Distribution:**")
                st.json(imb_info['class_distribution'])
            
            with col2:
                st.metric("Imbalance Ratio", f"{imb_info['imbalance_ratio']:.2f}:1")
            
            st.subheader("Suggested Fix")
            st.info('''
            **Options:**
            - Use SMOTE (Synthetic Minority Over-sampling)
            - Apply class weights in models
            - Undersample majority class
            - No action (some models handle imbalance well)
            ''')
            
            imbalance_action = st.radio(
                "How would you like to handle class imbalance?",
                ["No action", "Use SMOTE", "Apply class weights"],
                key="imbalance_action"
            )
            st.session_state.preprocessing_decisions['imbalance_handling'] = imbalance_action
    
    # High Cardinality 
    if len(issues['high_cardinality']) > 0:
        with st.expander("High Cardinality Features", expanded=True):
            st.write(f"**Found {len(issues['high_cardinality'])} high cardinality feature(s)**")
            
            cardinality_df = pd.DataFrame(issues['high_cardinality'])
            st.dataframe(cardinality_df, use_container_width=True)
            
            st.subheader("Suggested Fix")
            st.info('''
            **Options:**
            - Group rare categories into 'Other'
            - Use target encoding
            - Remove features with extreme cardinality
            ''')
            
            is_cardinality_fixed = st.checkbox("Group rare categories", key="fix_cardinality")
            
            if is_cardinality_fixed:
                st.session_state.preprocessing_decisions['fix_high_cardinality'] = True
                threshold = st.slider("Minimum frequency (%)", 1, 10, 5)
                st.session_state.preprocessing_decisions['cardinality_threshold'] = threshold / 100
            else:
                # FIX: Remove the keys if unchecked so it disappears from the summary table
                if 'fix_high_cardinality' in st.session_state.preprocessing_decisions:
                    del st.session_state.preprocessing_decisions['fix_high_cardinality']
                if 'cardinality_threshold' in st.session_state.preprocessing_decisions:
                    del st.session_state.preprocessing_decisions['cardinality_threshold']

    # Constant Features 
    if len(issues['constant_features']) > 0:
        with st.expander("Constant/Near-Constant Features", expanded=True):
            st.write(f"**Found {len(issues['constant_features'])} constant/near-constant feature(s)**")
            
            constant_df = pd.DataFrame(issues['constant_features'])
            st.dataframe(constant_df, use_container_width=True)
            
            st.subheader("Suggested Fix")
            st.warning("Constant features provide no information for prediction and should be removed.")
            
            is_constant_removed = st.checkbox("✅ Remove constant features", key="remove_constant")
            
            if is_constant_removed:
                st.session_state.preprocessing_decisions['remove_constant_features'] = True
            else:
                # FIX: Remove the key entirely if unchecked, so it disappears from the summary table
                if 'remove_constant_features' in st.session_state.preprocessing_decisions:
                    del st.session_state.preprocessing_decisions['remove_constant_features']
    
    # Duplicate Rows 
    if issues['duplicate_rows']['duplicate_count'] > 0:
        with st.expander("Duplicate Rows", expanded=True):
            dup_info = issues['duplicate_rows']
            st.write(f"**Found {dup_info['duplicate_count']} duplicate rows " 
                            f"({dup_info['duplicate_percentage']:.2f}%)**")
            
            st.subheader("Suggested Fix")
            st.warning("Duplicate rows can bias model training and should typically be removed.")
            
            is_duplicates_removed = st.checkbox("Remove duplicate rows", key="remove_duplicates")
            
            if is_duplicates_removed:
                st.session_state.preprocessing_decisions['remove_duplicates'] = True
            else:
                # FIX: Remove the key entirely if unchecked, so it disappears from the summary table
                if 'remove_duplicates' in st.session_state.preprocessing_decisions:
                    del st.session_state.preprocessing_decisions['remove_duplicates']
    
    # Summary
    st.markdown("---")
    st.subheader("Preprocessing Decisions Summary")
    
    if len(st.session_state.preprocessing_decisions) > 0:
        # Convert decisions into a DataFrame; convert values to strings to avoid Arrow warnings
        decisions_list = [
            {'Decision': k, 'Action': str(v)}
            for k, v in st.session_state.preprocessing_decisions.items()
        ]
        decisions_df = pd.DataFrame(decisions_list)
        st.dataframe(decisions_df, use_container_width=True)
        
        if st.button("Confirm Decisions", type="primary"):
            st.success("Decisions confirmed! Proceed to the Preprocessing step.")
    else:
        st.info("Select the fixes you want to apply, then confirm your decisions.")
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

def detect_issues(df, target_column):
    """
    Detect various data quality issues
    
    Args:
        df: pandas DataFrame
        target_column: name of the target column
        
    Returns:
        dict: Dictionary of detected issues
    """
    issues = {
        'missing_values': detect_missing_values(df),
        'outliers': detect_outliers(df, target_column),
        'class_imbalance': detect_class_imbalance(df, target_column),
        'high_cardinality': detect_high_cardinality(df, target_column),
        'constant_features': detect_constant_features(df, target_column),
        'duplicate_rows': detect_duplicate_rows(df)
    }
    
    # Note: df is passed but not strictly needed for display, 
    # included for consistency with the original function signature
    display_issues_and_get_approval(issues, df) 
    
    return issues

def detect_missing_values(df):
    """Detect missing values"""
    missing_info = []
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_pct = (missing_count / len(df)) * 100
            missing_info.append({
                'column': col,
                'missing_count': missing_count,
                'missing_percentage': missing_pct
            })
    
    return missing_info

def detect_outliers(df, target_column):
    """Detect outliers using IQR method"""
    outlier_info = []
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    
    for col in numerical_cols:
        data = df[col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        
        if outliers > 0:
            outlier_pct = (outliers / len(data)) * 100
            outlier_info.append({
                'column': col,
                'outlier_count': outliers,
                'outlier_percentage': outlier_pct,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
    
    return outlier_info

def detect_class_imbalance(df, target_column):
    """Detect class imbalance in target variable"""
    if target_column is None or target_column not in df.columns:
        return None
    
    class_counts = df[target_column].value_counts()
    total = len(df)
    
    imbalance_info = {
        'class_distribution': class_counts.to_dict(),
        'class_percentages': (class_counts / total * 100).to_dict(),
        'is_imbalanced': False,
        'imbalance_ratio': None
    }
    
    # Check if imbalanced (minority class < 20% of majority class)
    if not class_counts.empty and len(class_counts) > 1:
        max_count = class_counts.max()
        min_count = class_counts.min()
        
        if min_count / max_count < 0.2:
            imbalance_info['is_imbalanced'] = True
            imbalance_info['imbalance_ratio'] = max_count / min_count
    
    return imbalance_info

def detect_high_cardinality(df, target_column):
    """Detect high cardinality categorical features"""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    
    high_cardinality_features = []
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        unique_pct = (unique_count / len(df)) * 100
        
        # Flag if more than 50% unique or more than 20 unique categories
        if unique_pct > 50 or unique_count > 20:
            high_cardinality_features.append({
                'column': col,
                'unique_count': unique_count,
                'unique_percentage': unique_pct
            })
    
    return high_cardinality_features

def detect_constant_features(df, target_column):
    """Detect constant or near-constant features"""
    constant_features = []
    
    for col in df.columns:
        if col == target_column:
            continue
        
        # For numerical features
        if df[col].dtype in [np.int64, np.float64]:
            # Check if variance is very low
            if df[col].var() < 1e-7:
                constant_features.append({
                    'column': col,
                    'unique_values': df[col].nunique(),
                    'type': 'constant'
                })
        
        # For all features, check unique value ratio (excluding NaNs for ratio calculation simplicity)
        unique_ratio = df[col].nunique() / len(df.dropna(subset=[col])) if len(df.dropna(subset=[col])) > 0 else 0
        if unique_ratio < 0.01 and df[col].nunique() > 1: # Less than 1% unique, but has more than one value
            # Avoid duplicating features already flagged as 'constant'
            if not any(f['column'] == col and f['type'] == 'constant' for f in constant_features):
                constant_features.append({
                    'column': col,
                    'unique_values': df[col].nunique(),
                    'type': 'near-constant'
                })
    
    return constant_features

def detect_duplicate_rows(df):
    """Detect duplicate rows"""
    duplicate_count = df.duplicated().sum()
    
    return {
        'duplicate_count': duplicate_count,
        'duplicate_percentage': (duplicate_count / len(df)) * 100
    }

def display_issues_and_get_approval(issues, df):
    """Display detected issues and get user approval for fixes"""
    
    # Initialize session state for decisions if it doesn't exist
    if 'preprocessing_decisions' not in st.session_state:
        st.session_state.preprocessing_decisions = {}
        
    st.markdown("### Detected Issues")
    
    total_issues = sum([
        len(issues['missing_values']),
        len(issues['outliers']),
        1 if issues['class_imbalance'] and issues['class_imbalance']['is_imbalanced'] else 0,
        len(issues['high_cardinality']),
        len(issues['constant_features']),
        1 if issues['duplicate_rows']['duplicate_count'] > 0 else 0
    ])
    
    if total_issues == 0:
        st.success("No major issues detected! Your dataset looks good.")
        return
    
    st.warning(f"Found {total_issues} issue(s) in your dataset")
    
    # Missing Values
    if len(issues['missing_values']) > 0:
        with st.expander("Missing Values", expanded=True):
            st.write(f"**Found missing values in {len(issues['missing_values'])} column(s)**")
            
            missing_df = pd.DataFrame(issues['missing_values'])
            st.dataframe(missing_df, use_container_width=True)
            
            st.subheader("Suggested Fix")
            st.info("""
            **Options:**
            - Impute numerical features with mean/median
            - Impute categorical features with mode
            - Remove rows with missing values (if < 5%)
            """)
            
            # Get user approval
            is_missing_fixed = st.checkbox("I want to fix missing values", key="fix_missing")
            
            if is_missing_fixed:
                st.session_state.preprocessing_decisions['fix_missing_values'] = True
                st.success("Will handle missing values during preprocessing!")
            else:
                # FIX: Remove the key if unchecked so it disappears from the summary table
                if 'fix_missing_values' in st.session_state.preprocessing_decisions:
                    del st.session_state.preprocessing_decisions['fix_missing_values']
    
    # Outliers
    if len(issues['outliers']) > 0:
        with st.expander("Outliers", expanded=True):
            st.write(f"**Found outliers in {len(issues['outliers'])} column(s)**")
            
            outlier_df = pd.DataFrame(issues['outliers'])
            st.dataframe(outlier_df[['column', 'outlier_count', 'outlier_percentage']], 
                         use_container_width=True)
            
            st.subheader("Suggested Fix")
            st.info("""
            **Options:**
            - Cap outliers at IQR bounds (Winsorization)
            - Remove outliers (use with caution)
            - Keep outliers (they might be important)
            """)
            
            outlier_action = st.radio(
                "How would you like to handle outliers?",
                ["Keep outliers", "Cap outliers", "Remove outliers"],
                key="outlier_action"
            )
            st.session_state.preprocessing_decisions['outlier_handling'] = outlier_action
    
    # Class Imbalance
    if issues['class_imbalance'] and issues['class_imbalance']['is_imbalanced']:
        with st.expander("Class Imbalance", expanded=True):
            imb_info = issues['class_imbalance']
            st.write("**Imbalanced class distribution detected!**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Class Distribution:**")
                st.json(imb_info['class_distribution'])
            
            with col2:
                st.metric("Imbalance Ratio", f"{imb_info['imbalance_ratio']:.2f}:1")
            
            st.subheader("Suggested Fix")
            st.info("""
            **Options:**
            - Use SMOTE (Synthetic Minority Over-sampling)
            - Apply class weights in models
            - Undersample majority class
            - No action (some models handle imbalance well)
            """)
            
            imbalance_action = st.radio(
                "How would you like to handle class imbalance?",
                ["No action", "Use SMOTE", "Apply class weights"],
                key="imbalance_action"
            )
            st.session_state.preprocessing_decisions['imbalance_handling'] = imbalance_action
    
    # High Cardinality 
    if len(issues['high_cardinality']) > 0:
        with st.expander("High Cardinality Features", expanded=True):
            st.write(f"**Found {len(issues['high_cardinality'])} high cardinality feature(s)**")
            
            cardinality_df = pd.DataFrame(issues['high_cardinality'])
            st.dataframe(cardinality_df, use_container_width=True)
            
            st.subheader("Suggested Fix")
            st.info("""
            **Options:**
            - Group rare categories into 'Other'
            - Use target encoding
            - Remove features with extreme cardinality
            """)
            
            is_cardinality_fixed = st.checkbox("Group rare categories", key="fix_cardinality")
            
            if is_cardinality_fixed:
                st.session_state.preprocessing_decisions['fix_high_cardinality'] = True
                threshold = st.slider("Minimum frequency (%)", 1, 10, 5)
                st.session_state.preprocessing_decisions['cardinality_threshold'] = threshold / 100
            else:
                # FIX: Remove the keys if unchecked so it disappears from the summary table
                if 'fix_high_cardinality' in st.session_state.preprocessing_decisions:
                    del st.session_state.preprocessing_decisions['fix_high_cardinality']
                if 'cardinality_threshold' in st.session_state.preprocessing_decisions:
                    del st.session_state.preprocessing_decisions['cardinality_threshold']

    # Constant Features 
    if len(issues['constant_features']) > 0:
        with st.expander("Constant/Near-Constant Features", expanded=True):
            st.write(f"**Found {len(issues['constant_features'])} constant/near-constant feature(s)**")
            
            constant_df = pd.DataFrame(issues['constant_features'])
            st.dataframe(constant_df, use_container_width=True)
            
            st.subheader("Suggested Fix")
            st.warning("Constant features provide no information for prediction and should be removed.")
            
            is_constant_removed = st.checkbox("Remove constant features", key="remove_constant")
            
            if is_constant_removed:
                st.session_state.preprocessing_decisions['remove_constant_features'] = True
            else:
                # FIX: Remove the key entirely if unchecked, so it disappears from the summary table
                if 'remove_constant_features' in st.session_state.preprocessing_decisions:
                    del st.session_state.preprocessing_decisions['remove_constant_features']
    
    # Duplicate Rows 
    if issues['duplicate_rows']['duplicate_count'] > 0:
        with st.expander("Duplicate Rows", expanded=True):
            dup_info = issues['duplicate_rows']
            st.write(f"**Found {dup_info['duplicate_count']} duplicate rows " 
                            f"({dup_info['duplicate_percentage']:.2f}%)**")
            
            st.subheader("Suggested Fix")
            st.warning("Duplicate rows can bias model training and should typically be removed.")
            
            is_duplicates_removed = st.checkbox("Remove duplicate rows", key="remove_duplicates")
            
            if is_duplicates_removed:
                st.session_state.preprocessing_decisions['remove_duplicates'] = True
            else:
                # FIX: Remove the key entirely if unchecked, so it disappears from the summary table
                if 'remove_duplicates' in st.session_state.preprocessing_decisions:
                    del st.session_state.preprocessing_decisions['remove_duplicates']
    
    # Summary
    st.markdown("---")
    st.subheader("Preprocessing Decisions Summary")
    
    if len(st.session_state.preprocessing_decisions) > 0:
        # Convert decisions into a DataFrame; convert values to strings to avoid Arrow warnings
        decisions_list = [
            {'Decision': k, 'Action': str(v)}
            for k, v in st.session_state.preprocessing_decisions.items()
        ]
        decisions_df = pd.DataFrame(decisions_list)
        st.dataframe(decisions_df, use_container_width=True)
        
        if st.button("Confirm Decisions", type="primary"):
            st.success("Decisions confirmed! Proceed to the Preprocessing step.")
    else:
        st.info("Select the fixes you want to apply, then confirm your decisions.")