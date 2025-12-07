"""
Model Comparison Module
Compare and visualize model performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc

def compare_models(model_results):
    """
    Compare performance of all trained models
    
    Args:
        model_results: Dictionary of model results
    """
    st.subheader("Model Performance Comparison")
    
    if not model_results or len(model_results) == 0:
        st.warning("No models have been trained yet!")
        return
    
    # Create comparison table
    comparison_data = []
    
    for model_name, results in model_results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{results['accuracy']:.4f}",
            'Precision': f"{results['precision']:.4f}",
            'Recall': f"{results['recall']:.4f}",
            'F1-Score': f"{results['f1_score']:.4f}",
            'ROC-AUC': f"{results['roc_auc']:.4f}" if results['roc_auc'] is not None else 'N/A',
            'Training Time (s)': f"{results['training_time']:.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by F1-Score
    comparison_df['F1_numeric'] = comparison_df['F1-Score'].astype(float)
    comparison_df = comparison_df.sort_values('F1_numeric', ascending=False)
    comparison_df = comparison_df.drop('F1_numeric', axis=1)
    
    # Display comparison table
    st.subheader("Performance Metrics Comparison")
    st.dataframe(comparison_df, use_container_width=True)
    
    # Highlight best model
    best_model = comparison_df.iloc[0]['Model']
    st.success(f"🏆 **Best Model: {best_model}** (based on F1-Score)")
    
    # Download results
    csv = comparison_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="model_comparison_results.csv",
        mime="text/csv"
    )
    
    # Visualizations
    st.markdown("---")
    st.subheader("Performance Visualizations")
    
    # Create tabs for different visualizations
    tabs = st.tabs([
        "Metrics Comparison",
        "Training Time",
        "Confusion Matrices",
        "ROC Curves",
        "Detailed Reports"
    ])
    
    # Tab 1: Metrics Comparison
    with tabs[0]:
        plot_metrics_comparison(model_results)
    
    # Tab 2: Training Time
    with tabs[1]:
        plot_training_time(model_results)
    
    # Tab 3: Confusion Matrices
    with tabs[2]:
        plot_confusion_matrices(model_results)
    
    # Tab 4: ROC Curves
    with tabs[3]:
        plot_roc_curves(model_results)
    
    # Tab 5: Detailed Reports
    with tabs[4]:
        display_detailed_reports(model_results)

def plot_metrics_comparison(model_results):
    """Plot comparison of all metrics"""
    st.subheader("Metrics Comparison")
    
    # Prepare data
    models = list(model_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Create figure with subplots
    fig = go.Figure()
    
    for metric, metric_name in zip(metrics, metric_names):
        values = [model_results[model][metric] for model in models]
        
        fig.add_trace(go.Bar(
            name=metric_name,
            x=models,
            y=values,
            text=[f'{v:.3f}' for v in values],
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Model Performance Metrics Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart
    st.subheader("Radar Chart Comparison")
    
    selected_models = st.multiselect(
        "Select models to compare:",
        models,
        default=models[:min(3, len(models))]
    )
    
    if selected_models:
        fig = go.Figure()
        
        for model in selected_models:
            values = [
                model_results[model]['accuracy'],
                model_results[model]['precision'],
                model_results[model]['recall'],
                model_results[model]['f1_score']
            ]
            values.append(values[0])  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_names + [metric_names[0]],
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

def plot_training_time(model_results):
    """Plot training time comparison"""
    st.subheader("Training Time Comparison")
    
    models = list(model_results.keys())
    times = [model_results[model]['training_time'] for model in models]
    
    # Create bar chart
    fig = px.bar(
        x=models,
        y=times,
        title='Training Time by Model',
        labels={'x': 'Model', 'y': 'Time (seconds)'},
        text=[f'{t:.2f}s' for t in times],
        color=times,
        color_continuous_scale='Viridis'
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Fastest Model", models[np.argmin(times)], f"{min(times):.2f}s")
    
    with col2:
        st.metric("Slowest Model", models[np.argmax(times)], f"{max(times):.2f}s")
    
    with col3:
        st.metric("Average Time", "-", f"{np.mean(times):.2f}s")

def plot_confusion_matrices(model_results):
    """Plot confusion matrices for all models"""
    st.subheader("Confusion Matrices")
    
    models = list(model_results.keys())
    
    # Select model
    selected_model = st.selectbox("Select model:", models)
    
    if selected_model:
        cm = model_results[selected_model]['confusion_matrix']
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        ax.set_title(f'Confusion Matrix: {selected_model}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        st.pyplot(fig)
        plt.close()
        
        # Normalized confusion matrix
        st.subheader("Normalized Confusion Matrix")
        
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            ax=ax,
            cbar_kws={'label': 'Percentage'}
        )
        ax.set_title(f'Normalized Confusion Matrix: {selected_model}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        st.pyplot(fig)
        plt.close()
        
        # Calculate per-class metrics
        st.subheader("Per-Class Performance")
        
        n_classes = cm.shape[0]
        class_metrics = []
        
        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics.append({
                'Class': i,
                'Precision': f"{precision:.3f}",
                'Recall': f"{recall:.3f}",
                'F1-Score': f"{f1:.3f}",
                'Support': cm[i, :].sum()
            })
        
        class_metrics_df = pd.DataFrame(class_metrics)
        st.dataframe(class_metrics_df, use_container_width=True)

def plot_roc_curves(model_results):
    """Plot ROC curves for binary classification"""
    st.subheader("ROC Curves")
    
    # Check if binary classification
    first_result = list(model_results.values())[0]
    n_classes = len(np.unique(first_result['y_test']))
    
    if n_classes != 2:
        st.info("ROC curves are only available for binary classification problems.")
        st.write("For multiclass problems, consider using:")
        st.write("- One-vs-Rest (OvR) ROC curves")
        st.write("- Precision-Recall curves")
        st.write("- Confusion matrix analysis")
        return
    
    # Plot ROC curves for all models
    fig = go.Figure()
    
    for model_name, results in model_results.items():
        if results['roc_auc'] is not None:
            model = results['model']
            y_test = results['y_test']
            y_pred = results['y_pred']
            
            # Get predicted probabilities - use y_pred as fallback
            try:
                if hasattr(model, 'predict_proba') and 'X_test' in results:
                    # Use X_test stored in results
                    y_score = model.predict_proba(results['X_test'])[:, 1]
                else:
                    # Fallback to predictions
                    y_score = y_pred
            except:
                # If anything fails, use predictions
                y_score = y_pred
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = results['roc_auc']
            
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {roc_auc:.3f})',
                line=dict(width=2)
            ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=600,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ROC-AUC comparison
    st.subheader("ROC-AUC Scores")
    
    roc_data = []
    for model_name, results in model_results.items():
        if results['roc_auc'] is not None:
            roc_data.append({
                'Model': model_name,
                'ROC-AUC': results['roc_auc']
            })
    
    if roc_data:
        roc_df = pd.DataFrame(roc_data).sort_values('ROC-AUC', ascending=False)
        
        fig = px.bar(
            roc_df,
            x='Model',
            y='ROC-AUC',
            title='ROC-AUC Scores by Model',
            text='ROC-AUC',
            color='ROC-AUC',
            color_continuous_scale='Blues'
        )
        
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)

def display_detailed_reports(model_results):
    """Display detailed classification reports"""
    st.subheader("Detailed Classification Reports")
    
    models = list(model_results.keys())
    
    # Select model
    selected_model = st.selectbox("Select model for detailed report:", models, key="detailed_report")
    
    if selected_model:
        results = model_results[selected_model]
        
        # Model information
        st.markdown(f"### {selected_model}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.4f}")
        
        with col2:
            st.metric("Precision", f"{results['precision']:.4f}")
        
        with col3:
            st.metric("Recall", f"{results['recall']:.4f}")
        
        with col4:
            st.metric("F1-Score", f"{results['f1_score']:.4f}")
        
        # Best hyperparameters
        st.subheader("Best Hyperparameters")
        params_df = pd.DataFrame([
            {'Parameter': k, 'Value': str(v)}
            for k, v in results['best_params'].items()
        ])
        st.dataframe(params_df, use_container_width=True)
        
        # Classification report
        st.subheader("Classification Report")
        
        class_report = results['classification_report']
        
        # Convert to DataFrame
        report_data = []
        for label, metrics in class_report.items():
            if isinstance(metrics, dict) and label not in ['accuracy', 'macro avg', 'weighted avg']:
                report_data.append({
                    'Class': label,
                    'Precision': f"{metrics['precision']:.3f}",
                    'Recall': f"{metrics['recall']:.3f}",
                    'F1-Score': f"{metrics['f1-score']:.3f}",
                    'Support': metrics['support']
                })
        
        if report_data:
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df, use_container_width=True)
        
        # Macro and Weighted averages
        if 'macro avg' in class_report:
            st.subheader("Average Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Macro Average:**")
                macro = class_report['macro avg']
                st.write(f"- Precision: {macro['precision']:.3f}")
                st.write(f"- Recall: {macro['recall']:.3f}")
                st.write(f"- F1-Score: {macro['f1-score']:.3f}")
            
            with col2:
                st.write("**Weighted Average:**")
                weighted = class_report['weighted avg']
                st.write(f"- Precision: {weighted['precision']:.3f}")
                st.write(f"- Recall: {weighted['recall']:.3f}")
                st.write(f"- F1-Score: {weighted['f1-score']:.3f}")
