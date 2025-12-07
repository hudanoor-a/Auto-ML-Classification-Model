"""
Data Loading and Basic Information Display
"""

import streamlit as st
import pandas as pd
import numpy as np

def load_dataset(uploaded_file):
    """
    Load dataset from uploaded file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def display_dataset_info(df):
    """
    Display basic dataset information
    
    Args:
        df: pandas DataFrame
    """
    st.subheader("Dataset Overview")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    
    with col2:
        st.metric("Total Columns", len(df.columns))
    
    with col3:
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        st.metric("Numerical Features", len(numerical_cols))
    
    with col4:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        st.metric("Categorical Features", len(categorical_cols))
    
    # Dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Column types
    st.subheader("Column Types")
    col_types = pd.DataFrame({
        'Column': df.columns,
        'Data Type': [str(dtype) for dtype in df.dtypes.values],  # Convert to string
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(col_types, use_container_width=True)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    # Numerical columns
    if len(numerical_cols) > 0:
        st.write("**Numerical Features:**")
        st.dataframe(df[numerical_cols].describe(), use_container_width=True)
    
    # Categorical columns
    if len(categorical_cols) > 0:
        st.write("**Categorical Features:**")
        cat_summary = pd.DataFrame({
            'Column': categorical_cols,
            'Unique Values': [df[col].nunique() for col in categorical_cols],
            'Most Frequent': [df[col].mode()[0] if len(df[col].mode()) > 0 else None for col in categorical_cols],
            'Frequency': [df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0 for col in categorical_cols]
        })
        st.dataframe(cat_summary, use_container_width=True)
    
    # Memory usage
    st.subheader("Memory Usage")
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # Convert to MB
    st.info(f"Total Memory Usage: **{memory_usage:.2f} MB**")

def get_column_types(df):
    """
    Categorize columns by their data types
    
    Args:
        df: pandas DataFrame
        
    Returns:
        dict: Dictionary with 'numerical' and 'categorical' column lists
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return {
        'numerical': numerical_cols,
        'categorical': categorical_cols
    }
