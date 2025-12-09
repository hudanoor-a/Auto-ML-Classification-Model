"""
Model Comparison Module
Compare and visualize model performance with AUC-ROC and MCC metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from sklearn.preprocessing import label_binarize
from itertools import cycle, combinations

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
            'MCC': f"{results['mcc']:.4f}" if 'mcc' in results and results['mcc'] is not None else 'N/A',
            'Training Time (s)': f"{results['training_time']:.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Smart sorting with tiebreakers: F1-Score > ROC-AUC > MCC > Accuracy > Training Time
    def convert_metric(val):
        """Convert metric string to float, treating 'N/A' as -1"""
        return float(val) if val != 'N/A' else -1.0
    
    comparison_df['F1_numeric'] = comparison_df['F1-Score'].apply(convert_metric)
    comparison_df['ROC_AUC_numeric'] = comparison_df['ROC-AUC'].apply(convert_metric)
    comparison_df['MCC_numeric'] = comparison_df['MCC'].apply(convert_metric)
    comparison_df['Accuracy_numeric'] = comparison_df['Accuracy'].apply(convert_metric)
    comparison_df['Training_Time_numeric'] = comparison_df['Training Time (s)'].astype(float)
    
    # Sort by multiple criteria: F1 > ROC-AUC > MCC > Accuracy > Training Time (ascending for time)
    comparison_df = comparison_df.sort_values(
        by=['F1_numeric', 'ROC_AUC_numeric', 'MCC_numeric', 'Accuracy_numeric', 'Training_Time_numeric'],
        ascending=[False, False, False, False, True]  # True for training time (lower is better)
    )
    
    # Drop helper columns
    comparison_df = comparison_df.drop(['F1_numeric', 'ROC_AUC_numeric', 'MCC_numeric', 'Accuracy_numeric', 'Training_Time_numeric'], axis=1)
    
    # Display comparison table
    st.subheader("Performance Metrics Comparison")
    st.dataframe(comparison_df, use_container_width=True)
    
    # Highlight best model with detailed reasoning
    best_model = comparison_df.iloc[0]['Model']
    st.success(f"🏆 **Best Model: {best_model}**")
    
    # Show ranking explanation
    with st.expander("📊 Model Selection Criteria"):
        st.write("Models are ranked using the following tiebreaker hierarchy:")
        st.write("1. **F1-Score** (primary metric - balances precision and recall)")
        st.write("2. **ROC-AUC** (tiebreaker #1 - overall discrimination ability)")
        st.write("3. **MCC** (tiebreaker #2 - balanced measure for imbalanced data)")
        st.write("4. **Accuracy** (tiebreaker #3 - overall correctness)")
        st.write("5. **Training Time** (final tiebreaker - efficiency, lower is better)")
        st.write("")
        st.info("💡 If all performance metrics are equal, the faster model is preferred for production efficiency.")
    
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
        "Advanced Metrics",
        "Training Time",
        "Confusion Matrices",
        "ROC Curves",
        "Detailed Reports"
    ])
    
    # Tab 1: Metrics Comparison
    with tabs[0]:
        plot_metrics_comparison(model_results)
    
    # Tab 2: Advanced Metrics (ROC-AUC and MCC)
    with tabs[1]:
        plot_advanced_metrics(model_results)
    
    # Tab 3: Training Time
    with tabs[2]:
        plot_training_time(model_results)
    
    # Tab 4: Confusion Matrices
    with tabs[3]:
        plot_confusion_matrices(model_results)
    
    # Tab 5: ROC Curves
    with tabs[4]:
        plot_roc_curves(model_results)
    
    # Tab 6: Detailed Reports
    with tabs[5]:
        display_detailed_reports(model_results)

def plot_metrics_comparison(model_results):
    """Plot comparison of all metrics"""
    st.subheader("Core Metrics Comparison")
    
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

def plot_advanced_metrics(model_results):
    """Plot ROC-AUC and MCC metrics comparison"""
    st.subheader("Advanced Metrics: ROC-AUC and MCC")
    
    models = list(model_results.keys())
    
    # Prepare data for ROC-AUC and MCC
    roc_auc_values = []
    mcc_values = []
    valid_models = []
    
    for model in models:
        results = model_results[model]
        if results['roc_auc'] is not None:
            roc_auc_values.append(results['roc_auc'])
        else:
            roc_auc_values.append(None)
        
        if 'mcc' in results and results['mcc'] is not None:
            mcc_values.append(results['mcc'])
        else:
            mcc_values.append(None)
        
        valid_models.append(model)
    
    # Create side-by-side bar charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ROC-AUC Scores")
        valid_roc = [(m, v) for m, v in zip(valid_models, roc_auc_values) if v is not None]
        
        if valid_roc:
            models_roc, values_roc = zip(*valid_roc)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(models_roc),
                    y=list(values_roc),
                    text=[f'{v:.4f}' for v in values_roc],
                    textposition='outside',
                    marker=dict(
                        color=list(values_roc),
                        colorscale='Blues',
                        showscale=True,
                        colorbar=dict(title="ROC-AUC")
                    )
                )
            ])
            
            fig.update_layout(
                title='ROC-AUC by Model',
                xaxis_title='Model',
                yaxis_title='ROC-AUC Score',
                height=400,
                yaxis=dict(range=[0, 1.05])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Best ROC-AUC
            best_idx = np.argmax(values_roc)
            st.info(f"🥇 Best ROC-AUC: **{models_roc[best_idx]}** ({values_roc[best_idx]:.4f})")
        else:
            st.warning("ROC-AUC not available for these models")
    
    with col2:
        st.subheader("MCC Scores")
        valid_mcc = [(m, v) for m, v in zip(valid_models, mcc_values) if v is not None]
        
        if valid_mcc:
            models_mcc, values_mcc = zip(*valid_mcc)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(models_mcc),
                    y=list(values_mcc),
                    text=[f'{v:.4f}' for v in values_mcc],
                    textposition='outside',
                    marker=dict(
                        color=list(values_mcc),
                        colorscale='Greens',
                        showscale=True,
                        colorbar=dict(title="MCC")
                    )
                )
            ])
            
            fig.update_layout(
                title='Matthews Correlation Coefficient by Model',
                xaxis_title='Model',
                yaxis_title='MCC Score',
                height=400,
                yaxis=dict(range=[-1, 1.05])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Best MCC
            best_idx = np.argmax(values_mcc)
            st.info(f"🥇 Best MCC: **{models_mcc[best_idx]}** ({values_mcc[best_idx]:.4f})")
        else:
            st.warning("MCC not available for these models")
    
    # Combined scatter plot
    st.subheader("ROC-AUC vs MCC Comparison")
    
    valid_both = [(m, r, mc) for m, r, mc in zip(valid_models, roc_auc_values, mcc_values) 
                  if r is not None and mc is not None]
    
    if valid_both:
        models_both, roc_both, mcc_both = zip(*valid_both)
        
        fig = go.Figure(data=[
            go.Scatter(
                x=list(roc_both),
                y=list(mcc_both),
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=list(range(len(models_both))),
                    colorscale='Viridis',
                    showscale=False,
                    line=dict(width=2, color='white')
                ),
                text=list(models_both),
                textposition='top center',
                textfont=dict(size=10)
            )
        ])
        
        fig.update_layout(
            title='ROC-AUC vs MCC: Model Performance Map',
            xaxis_title='ROC-AUC',
            yaxis_title='MCC',
            height=500,
            xaxis=dict(range=[0, 1.05]),
            yaxis=dict(range=[-1, 1.05]),
            hovermode='closest'
        )
        
        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Interpretation:** Models in the upper-right quadrant (high ROC-AUC and high MCC) are the best performers.")
    
    # Metric descriptions
    with st.expander("📚 Understanding Advanced Metrics"):
        st.markdown("""
        ### ROC-AUC (Area Under the Receiver Operating Characteristic Curve)
        - **Range:** 0 to 1 (higher is better)
        - **Interpretation:** Measures the model's ability to distinguish between classes
        - **0.5:** Random classifier (no discrimination ability)
        - **1.0:** Perfect classifier
        - **Best for:** Binary classification, especially with imbalanced datasets
        
        ### MCC (Matthews Correlation Coefficient)
        - **Range:** -1 to +1
        - **Interpretation:** Balanced measure that considers all confusion matrix values
        - **+1:** Perfect prediction
        - **0:** Random prediction
        - **-1:** Complete disagreement between prediction and observation
        - **Best for:** Imbalanced datasets, provides reliable metric even with different class sizes
        - **Advantage:** Only produces high scores if the prediction performed well on all confusion matrix categories
        """)

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
    """Plot ROC curves for binary and multiclass classification"""
    st.subheader("ROC Curves")
    
    # Check if binary or multiclass classification
    first_result = list(model_results.values())[0]
    n_classes = len(np.unique(first_result['y_test']))
    
    if n_classes > 2:
        # Multiclass classification - offer OvR and OvO
        plot_multiclass_roc_curves(model_results, n_classes)
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

def plot_multiclass_roc_curves(model_results, n_classes):
    """Plot ROC curves for multiclass classification using OvR and OvO strategies"""
    
    st.info(f"📊 Detected {n_classes}-class classification problem. Choose a strategy for ROC analysis:")
    
    # Strategy selection
    strategy = st.radio(
        "Select ROC Strategy:",
        options=["One-vs-Rest (OvR)", "One-vs-One (OvO)", "Both"],
        help="""
        **One-vs-Rest (OvR)**: Each class is compared against all other classes combined.
        **One-vs-One (OvO)**: Each pair of classes is compared separately.
        """
    )
    
    # Model selection for multiclass
    models = list(model_results.keys())
    selected_models = st.multiselect(
        "Select models to visualize:",
        models,
        default=[models[0]] if models else []
    )
    
    if not selected_models:
        st.warning("Please select at least one model to visualize.")
        return
    
    # Display strategies
    if strategy in ["One-vs-Rest (OvR)", "Both"]:
        st.markdown("---")
        st.subheader("📈 One-vs-Rest (OvR) ROC Curves")
        plot_ovr_roc_curves(model_results, selected_models, n_classes)
    
    if strategy in ["One-vs-One (OvO)", "Both"]:
        st.markdown("---")
        st.subheader("📉 One-vs-One (OvO) ROC Curves")
        plot_ovo_roc_curves(model_results, selected_models, n_classes)

def plot_ovr_roc_curves(model_results, selected_models, n_classes):
    """Plot One-vs-Rest ROC curves for multiclass classification"""
    
    for model_name in selected_models:
        results = model_results[model_name]
        model = results['model']
        y_test = results['y_test']
        
        st.markdown(f"#### {model_name}")
        
        # Get predicted probabilities
        try:
            if hasattr(model, 'predict_proba') and 'X_test' in results:
                y_score = model.predict_proba(results['X_test'])
            else:
                st.warning(f"Model {model_name} does not support probability predictions. Skipping ROC curves.")
                continue
        except Exception as e:
            st.error(f"Error getting predictions for {model_name}: {str(e)}")
            continue
        
        # Binarize the labels for OvR
        classes = np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes)
        
        # If binary after binarization, reshape
        if y_test_bin.shape[1] == 1:
            y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # Plot all ROC curves
        fig = go.Figure()
        
        # Colors for each class
        colors = px.colors.qualitative.Set1[:n_classes]
        
        # Plot ROC curve for each class
        for i, color in zip(range(n_classes), colors):
            fig.add_trace(go.Scatter(
                x=fpr[i],
                y=tpr[i],
                mode='lines',
                name=f'Class {classes[i]} (AUC = {roc_auc[i]:.3f})',
                line=dict(color=color, width=2)
            ))
        
        # Plot micro-average ROC curve
        fig.add_trace(go.Scatter(
            x=fpr["micro"],
            y=tpr["micro"],
            mode='lines',
            name=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
            line=dict(color='deeppink', width=3, dash='dot')
        ))
        
        # Plot macro-average ROC curve
        fig.add_trace(go.Scatter(
            x=fpr["macro"],
            y=tpr["macro"],
            mode='lines',
            name=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
            line=dict(color='navy', width=3, dash='dash')
        ))
        
        # Add diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'One-vs-Rest ROC Curves - {model_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=600,
            hovermode='closest',
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display AUC scores table
        auc_data = []
        for i in range(n_classes):
            auc_data.append({
                'Class': f'Class {classes[i]}',
                'AUC Score': f'{roc_auc[i]:.4f}'
            })
        auc_data.append({'Class': 'Micro-average', 'AUC Score': f'{roc_auc["micro"]:.4f}'})
        auc_data.append({'Class': 'Macro-average', 'AUC Score': f'{roc_auc["macro"]:.4f}'})
        
        auc_df = pd.DataFrame(auc_data)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(auc_df, use_container_width=True)
        
        with col2:
            st.metric("Overall Macro-AUC", f"{roc_auc['macro']:.4f}")
            st.metric("Overall Micro-AUC", f"{roc_auc['micro']:.4f}")
        
        st.markdown("---")
    
    # Explanation
    with st.expander("ℹ️ Understanding One-vs-Rest ROC Curves"):
        st.markdown("""
        ### One-vs-Rest (OvR) Strategy
        
        **How it works:**
        - Each class is treated as the positive class
        - All other classes are combined into a negative class
        - A separate ROC curve is computed for each class
        
        **Metrics:**
        - **Per-class AUC**: Performance for each individual class
        - **Micro-average AUC**: Aggregates contributions of all classes (weights by class size)
        - **Macro-average AUC**: Simple average of per-class AUCs (treats all classes equally)
        
        **Interpretation:**
        - Higher AUC values (closer to 1.0) indicate better discrimination
        - Micro-average is more sensitive to majority classes
        - Macro-average treats all classes equally regardless of size
        """)

def plot_ovo_roc_curves(model_results, selected_models, n_classes):
    """Plot One-vs-One ROC curves for multiclass classification"""
    
    for model_name in selected_models:
        results = model_results[model_name]
        model = results['model']
        y_test = results['y_test']
        
        st.markdown(f"#### {model_name}")
        
        # Get predicted probabilities
        try:
            if hasattr(model, 'predict_proba') and 'X_test' in results:
                y_score = model.predict_proba(results['X_test'])
            else:
                st.warning(f"Model {model_name} does not support probability predictions. Skipping ROC curves.")
                continue
        except Exception as e:
            st.error(f"Error getting predictions for {model_name}: {str(e)}")
            continue
        
        classes = np.unique(y_test)
        n_classes_actual = len(classes)
        
        # Generate all pairwise combinations
        class_pairs = list(combinations(range(n_classes_actual), 2))
        
        # Calculate ROC curves for each pair
        pair_results = []
        
        for i, j in class_pairs:
            # Get indices for the two classes
            pair_mask = np.isin(y_test, [classes[i], classes[j]])
            
            if pair_mask.sum() == 0:
                continue
            
            y_test_pair = y_test[pair_mask]
            y_score_pair = y_score[pair_mask]
            
            # Binarize: class i = 0, class j = 1
            y_test_binary = (y_test_pair == classes[j]).astype(int)
            
            # Get probability for class j
            y_score_binary = y_score_pair[:, j] / (y_score_pair[:, i] + y_score_pair[:, j])
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test_binary, y_score_binary)
            roc_auc = auc(fpr, tpr)
            
            pair_results.append({
                'class_i': i,
                'class_j': j,
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc,
                'label': f'Class {classes[i]} vs Class {classes[j]}'
            })
        
        if not pair_results:
            st.warning(f"Could not compute OvO ROC curves for {model_name}")
            continue
        
        # Create visualization options
        viz_option = st.radio(
            f"Visualization for {model_name}:",
            options=["All Pairs", "Individual Pairs"],
            key=f"ovo_viz_{model_name}"
        )
        
        if viz_option == "All Pairs":
            # Plot all pairs together
            fig = go.Figure()
            
            colors = cycle(px.colors.qualitative.Set2)
            
            for pair, color in zip(pair_results, colors):
                fig.add_trace(go.Scatter(
                    x=pair['fpr'],
                    y=pair['tpr'],
                    mode='lines',
                    name=f"{pair['label']} (AUC = {pair['auc']:.3f})",
                    line=dict(color=color, width=2)
                ))
            
            # Add diagonal reference line
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='gray', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f'One-vs-One ROC Curves - {model_name}',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=600,
                hovermode='closest',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Individual pair selection
            pair_labels = [p['label'] for p in pair_results]
            selected_pair = st.selectbox(
                f"Select pair for {model_name}:",
                pair_labels,
                key=f"ovo_pair_{model_name}"
            )
            
            selected_pair_idx = pair_labels.index(selected_pair)
            pair = pair_results[selected_pair_idx]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=pair['fpr'],
                y=pair['tpr'],
                mode='lines',
                name=f"{pair['label']} (AUC = {pair['auc']:.3f})",
                line=dict(color='blue', width=3),
                fill='tonexty'
            ))
            
            # Add diagonal reference line
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='gray', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f'ROC Curve: {pair["label"]} - {model_name}',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=600,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics for this pair
            col1, col2 = st.columns(2)
            with col1:
                st.metric("AUC Score", f"{pair['auc']:.4f}")
            with col2:
                st.metric("Number of Samples", len(y_test[np.isin(y_test, [classes[pair['class_i']], classes[pair['class_j']]])]))
        
        # Display summary table of all pairs
        st.subheader("Pairwise AUC Scores")
        
        auc_summary = pd.DataFrame([
            {
                'Class Pair': p['label'],
                'AUC Score': f"{p['auc']:.4f}"
            }
            for p in pair_results
        ]).sort_values('AUC Score', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(auc_summary, use_container_width=True)
        
        with col2:
            avg_auc = np.mean([p['auc'] for p in pair_results])
            st.metric("Average Pairwise AUC", f"{avg_auc:.4f}")
            st.metric("Number of Pairs", len(pair_results))
        
        # Heatmap of pairwise AUC scores
        st.subheader("Pairwise AUC Heatmap")
        
        auc_matrix = np.zeros((n_classes_actual, n_classes_actual))
        for pair in pair_results:
            i, j = pair['class_i'], pair['class_j']
            auc_matrix[i, j] = pair['auc']
            auc_matrix[j, i] = pair['auc']  # Symmetric
        
        # Set diagonal to NaN for better visualization
        np.fill_diagonal(auc_matrix, np.nan)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        mask = np.isnan(auc_matrix)
        sns.heatmap(
            auc_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0.5,
            vmin=0,
            vmax=1,
            mask=mask,
            ax=ax,
            cbar_kws={'label': 'AUC Score'},
            xticklabels=[f'Class {c}' for c in classes],
            yticklabels=[f'Class {c}' for c in classes]
        )
        
        ax.set_title(f'Pairwise AUC Scores - {model_name}', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
    
    # Explanation
    with st.expander("ℹ️ Understanding One-vs-One ROC Curves"):
        st.markdown("""
        ### One-vs-One (OvO) Strategy
        
        **How it works:**
        - Creates a binary classifier for every pair of classes
        - For N classes, creates N×(N-1)/2 pairwise comparisons
        - Each ROC curve shows discrimination between two specific classes
        
        **Advantages:**
        - More detailed analysis of class separability
        - Can identify which class pairs are difficult to distinguish
        - Less affected by class imbalance
        
        **Interpretation:**
        - Each curve shows how well the model distinguishes between two specific classes
        - Lower AUC scores indicate classes that are harder to separate
        - The heatmap provides a quick overview of all pairwise comparisons
        - Average pairwise AUC gives an overall measure of multiclass discrimination
        
        **Use Cases:**
        - When you need to understand specific class confusions
        - For imbalanced multiclass problems
        - To identify which classes need more training data or feature engineering
        """)

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
        
        # Create metrics layout
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.4f}")
        
        with col2:
            st.metric("Precision", f"{results['precision']:.4f}")
        
        with col3:
            st.metric("Recall", f"{results['recall']:.4f}")
        
        with col4:
            st.metric("F1-Score", f"{results['f1_score']:.4f}")
        
        with col5:
            if results['roc_auc'] is not None:
                st.metric("ROC-AUC", f"{results['roc_auc']:.4f}")
            else:
                st.metric("ROC-AUC", "N/A")
        
        # Second row for MCC and training time
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'mcc' in results and results['mcc'] is not None:
                st.metric("MCC", f"{results['mcc']:.4f}")
            else:
                st.metric("MCC", "N/A")
        
        with col2:
            st.metric("Training Time", f"{results['training_time']:.2f}s")
        
        with col3:
            st.metric("Test Samples", len(results['y_test']))
        
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