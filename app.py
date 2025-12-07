"""
AutoML System for Classification - Streamlit Application
Main entry point for the application
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AutoML Classification System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom modules
from utils.data_loader import load_dataset, display_dataset_info
from utils.eda import perform_eda
from utils.issue_detector import detect_issues
from utils.preprocessor import preprocess_data
from utils.model_trainer import train_models
from utils.model_comparison import compare_models
from utils.report_generator import generate_report

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'preprocessing_decisions' not in st.session_state:
    st.session_state.preprocessing_decisions = {}
if 'detected_issues' not in st.session_state:
    st.session_state.detected_issues = {}

def main():
    """Main application function"""
    
    # Initialize step tracking
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    
    # Define steps
    steps = [
        {"name": "📁 Dataset Upload", "key": "dataset"},
        {"name": "📊 Exploratory Data Analysis", "key": "eda"},
        {"name": "⚠️ Issue Detection & Fixing", "key": "issues"},
        {"name": "⚙️ Preprocessing", "key": "preprocessing"},
        {"name": "🎯 Model Training", "key": "training"},
        {"name": "📈 Model Comparison", "key": "comparison"},
        {"name": "📄 Final Report", "key": "report"}
    ]
    
    # Header: show only on the first step
    if st.session_state.get('current_step', 0) == 0:
        st.title("🤖 AutoML Classification System")
        st.markdown("""
        Welcome to the **AutoML Classification System**! This intelligent system helps you:
        - 📊 Analyze your dataset automatically
        - 🔍 Detect and fix data quality issues
        - 🎯 Train multiple classification models
        - 📈 Compare model performance
        - 📄 Generate comprehensive reports
        """)
    
    # Progress indicator
    st.markdown("---")
    st.subheader("Workflow Progress")
    
    # Create progress bar
    progress_cols = st.columns(len(steps))
    for idx, (col, step) in enumerate(zip(progress_cols, steps)):
        with col:
            if idx < st.session_state.current_step:
                st.markdown(f"✅ **{step['name'].split()[1]}**")
            elif idx == st.session_state.current_step:
                st.markdown(f"▶️ **{step['name'].split()[1]}**")
            else:
                st.markdown(f"⭕ {step['name'].split()[1]}")
    
    progress_percentage = (st.session_state.current_step / len(steps)) * 100
    st.progress(progress_percentage / 100)
    
    st.markdown("---")
    
    # Sidebar - Manual navigation (optional)
    # Put the app logo/title at the top of the sidebar and keep it visible
    st.sidebar.markdown(
        """
        <div style="
            position: sticky;
            top: 0;
            background-color: rgba(255,255,255,0.98);
            # padding: 8px 12px;
            z-index: 9999;
            border-bottom: 1px solid #eee;
        ">
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="font-size:50px;">🤖</span>
                <div>
                    <div style="font-weight:700;font-size:38px;margin-bottom:0;">AutoML</div>
                    <div style="font-size:15px;color:#666;margin-top:2px;">Classification System</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("------------------")
    st.sidebar.subheader("Quick Navigation")

    st.sidebar.markdown("*You can jump to any completed step*")
    
    manual_step = st.sidebar.selectbox(
        "Jump to step:",
        options=range(len(steps)),
        format_func=lambda x: steps[x]['name'],
        index=st.session_state.current_step
    )
    
    if st.sidebar.button("Go to Selected Step"):
        # Only allow going back or to current step
        if manual_step <= st.session_state.current_step:
            st.session_state.current_step = manual_step
            st.rerun()
        else:
            st.sidebar.error("Please complete previous steps first!")
    
    # Display current step
    current_step_idx = st.session_state.current_step
    current_step = steps[current_step_idx]
    
    st.header(f"{current_step['name']}")
    
    # Step 0: Dataset Upload (persisted view if dataset already uploaded)
    if current_step_idx == 0:
        # If a dataset was previously uploaded, show persisted view
        df = st.session_state.get('dataset')

        def _clear_dataset():
            # Clear dataset + related state but keep navigation state
            keys_to_clear = ['dataset', 'processed_data', 'model_results', 'target_column', 'preprocessing_decisions', 'detected_issues']
            for k in keys_to_clear:
                if k in st.session_state:
                    del st.session_state[k]
            st.success('Dataset removed.')
            st.rerun()

        if df is not None:
            st.success("Dataset is currently loaded (persisted in session state).")
            col_info, col_actions = st.columns([4, 1])
            with col_info:
                display_dataset_info(df)
                # Target column selection (remember previous selection if present)
                st.subheader("Select Target Column")
                options = df.columns.tolist()
                default_index = 0
                if st.session_state.get('target_column') in options:
                    try:
                        default_index = options.index(st.session_state.get('target_column'))
                    except Exception:
                        default_index = 0
                target_col = st.selectbox("Choose the target variable for classification:", options=options, index=default_index, key='persisted_target_selector')
                st.session_state.target_column = target_col
                if target_col:
                    st.info(f"Target column set to: **{target_col}**")
                    st.subheader("Class Distribution")
                    class_counts = df[target_col].value_counts()
                    c1, c2 = st.columns(2)
                    with c1:
                        st.dataframe(class_counts)
                    with c2:
                        st.bar_chart(class_counts)

            with col_actions:
                # Provide a clear/remove dataset button
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Remove Dataset", use_container_width=True):
                    _clear_dataset()

            st.markdown("---")
            if st.button("Continue to EDA  ➡️", type="primary", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
        else:
            # No persisted dataset; show uploader
            uploaded_file = st.file_uploader(
                "Upload your CSV file",
                type=['csv'],
                help="Upload a CSV file containing your classification dataset"
            )

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.dataset = df
                    st.success("Dataset loaded successfully!")
                    display_dataset_info(df)
                    # Let user choose target column immediately after upload
                    options = df.columns.tolist()
                    default_index = 0
                    if st.session_state.get('target_column') in options:
                        try:
                            default_index = options.index(st.session_state.get('target_column'))
                        except Exception:
                            default_index = 0
                    target_col = st.selectbox("Choose the target variable for classification:", options=options, index=default_index, key='uploaded_target_selector')
                    st.session_state.target_column = target_col
                    if target_col:
                        st.info(f"Target column set to: **{target_col}**")
                        st.subheader("Class Distribution")
                        class_counts = df[target_col].value_counts()
                        c1, c2 = st.columns(2)
                        with c1:
                            st.dataframe(class_counts)
                        with c2:
                            st.bar_chart(class_counts)
                    st.markdown("---")
                    if st.button("Continue to EDA  ➡️", type="primary", use_container_width=True):
                        st.session_state.current_step = 1
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
            else:
                # Sample dataset option
                st.info("Don't have a dataset? Try our sample datasets!")
                sample_dataset = st.selectbox(
                    "Choose a sample dataset:",
                    ["None", "Iris", "Wine Quality"]
                )
                if sample_dataset != "None":
                    if st.button("Load Sample Dataset"):
                        df = load_sample_dataset(sample_dataset)
                        st.session_state.dataset = df
                        st.success(f"{sample_dataset} dataset loaded!")
                        st.rerun()
    
    # Step 1: EDA
    elif current_step_idx == 1:
        if st.session_state.dataset is None:
            st.warning("Please upload a dataset first!")
            if st.button("⬅️  Go Back to Dataset Upload"):
                st.session_state.current_step = 0
                st.rerun()
            return
        
        perform_eda(st.session_state.dataset, st.session_state.target_column)
        
        # Navigation buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️  Back to Dataset Upload", use_container_width=True):
                st.session_state.current_step = 0
                st.rerun()
        with col2:
            if st.button("Continue to Issue Detection  ➡️", type="primary", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
    
    # Step 2: Issue Detection
    elif current_step_idx == 2:
        if st.session_state.dataset is None:
            st.warning("Please upload a dataset first!")
            return
        
        detected_issues = detect_issues(
            st.session_state.dataset,
            st.session_state.target_column
        )
        st.session_state.detected_issues = detected_issues
        
        # Navigation buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️  Back to EDA", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
        with col2:
            if st.button("Continue to Preprocessing  ➡️", type="primary", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()
    
    # Step 3: Preprocessing
    elif current_step_idx == 3:
        if st.session_state.dataset is None:
            st.warning("Please upload a dataset first!")
            return
        
        processed_data = preprocess_data(
            st.session_state.dataset,
            st.session_state.target_column,
            st.session_state.detected_issues
        )
        
        if processed_data is not None:
            st.session_state.processed_data = processed_data
            
            # Navigation buttons
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅️  Back to Issue Detection", use_container_width=True):
                    st.session_state.current_step = 2
                    st.rerun()
            with col2:
                if st.button("Continue to Model Training  ➡️", type="primary", use_container_width=True):
                    st.session_state.current_step = 4
                    st.success("Preprocessing completed! Moving to model training...")
                    st.rerun()
    
    # Step 4: Model Training
    elif current_step_idx == 4:
        if st.session_state.processed_data is None:
            st.warning("Please complete preprocessing first!")
            return
        
        # Show training results if already trained
        if st.session_state.model_results is not None and len(st.session_state.model_results) > 0:
            st.success(f"Training completed! {len(st.session_state.model_results)} models trained successfully.")
            
            # Show brief summary
            st.subheader("Quick Summary")
            summary_data = []
            for model_name, results in st.session_state.model_results.items():
                summary_data.append({
                    'Model': model_name,
                    'Accuracy': f"{results['accuracy']:.3f}",
                    'F1-Score': f"{results['f1_score']:.3f}",
                    'Training Time': f"{results['training_time']:.2f}s"
                })
            
            summary_df = pd.DataFrame(summary_data).sort_values('F1-Score', ascending=False)
            st.dataframe(summary_df, use_container_width=True)
            
            # Individual model details in expanders
            st.subheader("Individual Model Details")
            st.markdown("*Click on each model to see detailed metrics and training time*")
            
            for model_name, results in st.session_state.model_results.items():
                with st.expander(f"{model_name} - {results['training_time']:.2f}s"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{results['accuracy']:.4f}")
                    with col2:
                        st.metric("Precision", f"{results['precision']:.4f}")
                    with col3:
                        st.metric("Recall", f"{results['recall']:.4f}")
                    with col4:
                        st.metric("F1-Score", f"{results['f1_score']:.4f}")
                    
                    st.write(f"**Training Time:** {results['training_time']:.2f} seconds")
                    if results['roc_auc'] is not None:
                        st.write(f"**ROC-AUC:** {results['roc_auc']:.4f}")
            
            # Navigation buttons AFTER training
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅️  Back to Preprocessing", use_container_width=True):
                    st.session_state.current_step = 3
                    st.rerun()
                if st.button("🔄 Retrain Models", use_container_width=True):
                    st.session_state.model_results = None
                    st.rerun()
            with col2:
                if st.button("View Model Comparison  ➡️", type="primary", use_container_width=True):
                    st.session_state.current_step = 5
                    st.rerun()
        else:
            # Training interface
            model_results = train_models(st.session_state.processed_data)
            
            # Save results immediately if returned
            if model_results is not None and len(model_results) > 0:
                st.session_state.model_results = model_results
                st.success(f"Successfully trained {len(model_results)} models!")
                st.rerun()  # Reload to show the results
            else:
                # Show back button if training not started
                st.markdown("---")
                if st.button("⬅️  Back to Preprocessing", use_container_width=True):
                    st.session_state.current_step = 3
                    st.rerun()
    
    # Step 5: Model Comparison
    elif current_step_idx == 5:
        if st.session_state.model_results is None:
            st.warning("Please train models first!")
            return
        
        compare_models(st.session_state.model_results)
        
        # Navigation buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️  Back to Model Training", use_container_width=True):
                st.session_state.current_step = 4
                st.rerun()
        with col2:
            if st.button("Generate Final Report  ➡️ ", type="primary", use_container_width=True):
                st.session_state.current_step = 6
                st.rerun()
    
    # Step 6: Final Report
    elif current_step_idx == 6:
        if st.session_state.model_results is None:
            st.warning("⚠️ Please complete all previous steps first!")
            return
        
        generate_report(
            st.session_state.dataset,
            st.session_state.target_column,
            st.session_state.detected_issues,
            st.session_state.preprocessing_decisions,
            st.session_state.model_results
        )
        
        # Navigation buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Model Comparison", use_container_width=True):
                st.session_state.current_step = 5
                st.rerun()
        with col2:
            if st.button("🔄 Start New Analysis", type="primary", use_container_width=True):
                # Reset all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    # # Footer
    # st.sidebar.markdown("---")
    # st.sidebar.markdown("### Session Status")
    # st.sidebar.write(f"Dataset: {'Loaded' if st.session_state.dataset is not None else 'Not loaded'}")
    # st.sidebar.write(f"Target: {st.session_state.target_column if st.session_state.target_column else 'Not selected'}")
    # st.sidebar.write(f"Preprocessing: {'Done' if st.session_state.processed_data is not None else 'Pending'}")
    # st.sidebar.write(f"Models: {'Trained' if st.session_state.model_results is not None else 'Not trained'}")
    
    # st.sidebar.markdown("---")
    # st.sidebar.markdown("### 📚 About")
    # st.sidebar.info("""
    # **AutoML Classification System**
    
    # Built with ❤️ using Streamlit
    
    # Features:
    # - Automated EDA
    # - Smart issue detection
    # - Multi-model training
    # - Hyperparameter optimization
    # - Comprehensive reporting
    # """)

def load_sample_dataset(dataset_name):
    """Load sample datasets"""
    from sklearn.datasets import load_iris, load_wine
    
    if dataset_name == "Iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        st.session_state.target_column = 'target'
    elif dataset_name == "Wine Quality":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        st.session_state.target_column = 'target'
    else:
        df = pd.DataFrame()
    
    return df

if __name__ == "__main__":
    main()