"""
Model Training Module
Train multiple classification models with hyperparameter optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report,
    matthews_corrcoef  # ADDED: MCC metric
)
import plotly.graph_objects as go
import plotly.express as px
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import label_binarize  # ADDED: For multiclass ROC
import warnings
warnings.filterwarnings('ignore')

# Simple rule-based classifier
class RuleBasedClassifier:
    """Simple rule-based classifier using decision rules"""
    
    def __init__(self):
        self.rules = []
        self.default_class = None
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility"""
        return {}
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility"""
        return self
    
    def fit(self, X, y):
        """Fit the rule-based classifier"""
        # For simplicity, use the most common class as default
        self.default_class = pd.Series(y).mode()[0]
        
        # Create simple rules based on feature means
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X
        
        # Generate rules based on feature statistics
        for col in range(min(3, X_df.shape[1])):  # Use up to 3 features
            threshold = X_df.iloc[:, col].median()
            self.rules.append({
                'feature': col,
                'threshold': threshold,
                'class': self.default_class
            })
        
        return self
    
    def predict(self, X):
        """Predict using simple rules"""
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X
        
        predictions = np.full(len(X_df), self.default_class)
        
        # Apply rules
        for rule in self.rules:
            feature = rule['feature']
            if feature < X_df.shape[1]:
                mask = X_df.iloc[:, feature] > rule['threshold']
                predictions[mask] = rule['class']
        
        return predictions
    
    def predict_proba(self, X):
        """Predict probabilities (simplified)"""
        preds = self.predict(X)
        n_classes = len(np.unique(preds))
        proba = np.zeros((len(preds), max(2, n_classes)))
        
        for i, pred in enumerate(preds):
            proba[i, int(pred)] = 0.8  # Assign high probability to predicted class
            # Distribute remaining probability
            remaining = 0.2 / (max(2, n_classes) - 1)
            for j in range(max(2, n_classes)):
                if j != int(pred):
                    proba[i, j] = remaining
        
        return proba

def train_models(processed_data):
    """
    Train multiple classification models with hyperparameter optimization
    
    Args:
        processed_data: Dictionary containing train/test splits
        
    Returns:
        dict: Results for all trained models
    """
    st.subheader("Model Training Configuration")
    
    # Extract data
    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    n_classes = processed_data['n_classes']
    imbalance_handling = processed_data.get('imbalance_handling', 'No action')
    
    # Critical validation: Check for NaN values
    # Convert to DataFrame if numpy array for easier checking
    if isinstance(X_train, np.ndarray):
        X_train_df = pd.DataFrame(X_train)
        X_test_df = pd.DataFrame(X_test)
    else:
        X_train_df = X_train
        X_test_df = X_test
    
    # Check for NaN values
    if X_train_df.isnull().any().any():
        st.error("Training data contains missing values!")
        st.warning("Applying automatic imputation to fix missing values...")
        
        # Force imputation
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train_df)
        X_test = imputer.transform(X_test_df)
        
        st.success("Missing values imputed automatically!")
    
    # Ensure no NaN in target
    if pd.Series(y_train).isnull().any():
        st.error("Target variable contains missing values! Cannot proceed.")
        st.info("Please go back to preprocessing and ensure missing values are handled.")
        return None
    
    # Calculate appropriate CV folds based on data
    # CV requires at least n_splits samples per class
    class_counts = pd.Series(y_train).value_counts()
    min_class_count = class_counts.min()
    max_cv_folds = min(10, min_class_count)  # Can't have more folds than smallest class
    
    # Hyperparameter optimization settings
    st.write("**Hyperparameter Optimization Settings:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        optimization_method = st.selectbox(
            "Optimization Method:",
            ["Grid Search", "Randomized Search", "No Optimization"]
        )
    
    with col2:
        if max_cv_folds < 2:
            st.warning(f"Dataset too small for cross-validation. Using simple train/test split.")
            cv_folds = 2
            optimization_method = "No Optimization"  # Force no optimization
        else:
            default_cv = min(5, max_cv_folds)
            cv_folds = st.slider(
                "Cross-Validation Folds:", 
                2, 
                max_cv_folds, 
                default_cv,
                help=f"Maximum {max_cv_folds} folds possible (limited by smallest class size: {min_class_count})"
            )
    
    # Model selection
    st.write("**Select Models to Train:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_lr = st.checkbox("Logistic Regression", value=True)
        train_knn = st.checkbox("K-Nearest Neighbors", value=True)
        train_dt = st.checkbox("Decision Tree", value=True)
        train_nb = st.checkbox("Naive Bayes", value=True)
    
    with col2:
        train_rf = st.checkbox("Random Forest", value=True)
        train_svm = st.checkbox("Support Vector Machine", value=True)
        train_rule = st.checkbox("Rule-Based Classifier", value=True)
    
    # Class weights
    use_class_weights = imbalance_handling == 'Apply class weights'
    
    if st.button("Start Training", type="primary"):
        model_results = {}
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        models_to_train = [
            ('Logistic Regression', train_lr),
            ('K-Nearest Neighbors', train_knn),
            ('Decision Tree', train_dt),
            ('Naive Bayes', train_nb),
            ('Random Forest', train_rf),
            ('Support Vector Machine', train_svm),
            ('Rule-Based Classifier', train_rule)
        ]
        
        total_models = sum([1 for _, train in models_to_train if train])
        current_model = 0
        
        # Train each selected model
        for model_name, should_train in models_to_train:
            if not should_train:
                continue
            
            current_model += 1
            status_text.write(f"Training {model_name}... ({current_model}/{total_models})")
            progress_bar.progress(current_model / total_models)
            
            try:
                result = train_single_model(
                    model_name,
                    X_train, X_test, y_train, y_test,
                    optimization_method,
                    cv_folds,
                    use_class_weights,
                    n_classes
                )
                
                model_results[model_name] = result
                
            except Exception as e:
                st.error(f"Error training {model_name}: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        
        status_text.write("Training completed!")
        
        return model_results
    
    return None

def train_single_model(model_name, X_train, X_test, y_train, y_test, 
                      optimization_method, cv_folds, use_class_weights, n_classes):
    """Train a single model with hyperparameter optimization"""
    
    start_time = time.time()
    
    # Get model and parameter grid
    model, param_grid = get_model_and_params(model_name, use_class_weights)
    
    # Hyperparameter optimization
    if optimization_method != "No Optimization" and param_grid:
        if optimization_method == "Grid Search":
            search = GridSearchCV(
                model, param_grid, cv=cv_folds, scoring='f1_weighted',
                n_jobs=-1, verbose=0
            )
        else:  # Randomized Search
            search = RandomizedSearchCV(
                model, param_grid, cv=cv_folds, scoring='f1_weighted',
                n_iter=10, n_jobs=-1, verbose=0, random_state=42
            )
        
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
    else:
        best_model = model
        best_model.fit(X_train, y_train)
        best_params = model.get_params()
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    training_time = time.time() - start_time
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ============ ADDED: Matthews Correlation Coefficient (MCC) ============
    try:
        mcc = matthews_corrcoef(y_test, y_pred)
    except Exception as e:
        st.warning(f"Could not calculate MCC for {model_name}: {str(e)}")
        mcc = None
    
    # ============ ENHANCED: ROC-AUC for both binary and multiclass ============
    roc_auc = None
    
    if n_classes == 2:
        # Binary classification
        try:
            if hasattr(best_model, 'predict_proba'):
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = y_pred
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except Exception as e:
            st.warning(f"Could not calculate ROC-AUC for {model_name}: {str(e)}")
            roc_auc = None
    
    elif n_classes > 2:
        # Multiclass classification - calculate macro-average ROC-AUC
        try:
            if hasattr(best_model, 'predict_proba'):
                y_pred_proba = best_model.predict_proba(X_test)
                
                # Use One-vs-Rest (OvR) macro-average
                roc_auc = roc_auc_score(
                    y_test, 
                    y_pred_proba, 
                    multi_class='ovr',
                    average='macro'
                )
            else:
                # Model doesn't support probabilities
                roc_auc = None
        except Exception as e:
            st.warning(f"Could not calculate multiclass ROC-AUC for {model_name}: {str(e)}")
            roc_auc = None
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # ============ ADDED: Store X_test for multiclass ROC curve plotting ============
    return {
        'model': best_model,
        'best_params': best_params,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,  # Now supports both binary and multiclass
        'mcc': mcc,  # ADDED: Matthews Correlation Coefficient
        'training_time': training_time,
        'classification_report': class_report,
        'y_pred': y_pred,
        'y_test': y_test,
        'X_test': X_test,  # ADDED: Required for multiclass ROC curve plotting
        'n_classes': n_classes  # ADDED: Store number of classes for reference
    }

def get_model_and_params(model_name, use_class_weights):
    """Get model and hyperparameter grid"""
    
    class_weight = 'balanced' if use_class_weights else None
    
    if model_name == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=42)
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        }
    
    elif model_name == 'K-Nearest Neighbors':
        model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier(class_weight=class_weight, random_state=42)
        param_grid = {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
    
    elif model_name == 'Naive Bayes':
        model = GaussianNB()
        param_grid = {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(class_weight=class_weight, random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2']
        }
    
    elif model_name == 'Support Vector Machine':
        # IMPORTANT: Enable probability=True for ROC curves
        model = SVC(class_weight=class_weight, random_state=42, probability=True)
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    
    elif model_name == 'Rule-Based Classifier':
        model = RuleBasedClassifier()
        param_grid = None  # No hyperparameters for rule-based
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, param_grid