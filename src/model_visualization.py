import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score
import os

def setup_visualization():
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('viridis')
    os.makedirs('../output/figures', exist_ok=True)

def visualize_classification_results():
    results = pd.read_csv('../output/classification_results.csv')
    feature_importance = pd.read_csv('../output/classification_feature_importance.csv')

    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(results['y_true'], results['rf_prob'])
    auc_rf = roc_auc_score(results['y_true'], results['rf_prob'])
    
    plt.plot(fpr, tpr, lw=2, label=f'Random Forest (AUC = {auc_rf:.3f})')
    
    fpr, tpr, _ = roc_curve(results['y_true'], results['lr_prob'])
    auc_lr = roc_auc_score(results['y_true'], results['lr_prob'])
    
    plt.plot(fpr, tpr, lw=2, linestyle='--', label=f'Logistic Regression (AUC = {auc_lr:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)

    thresholds = [0.3, 0.5, 0.7]
    for thresh in thresholds:
        y_pred = (results['rf_prob'] >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(results['y_true'], y_pred).ravel()
        current_fpr = fp / (fp + tn)
        current_tpr = tp / (tp + fn)
        plt.scatter(current_fpr, current_tpr, marker='o', s=50, 
                   label=f'Threshold = {thresh:.1f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve for Policy Copying Prediction', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('../output/figures/enhanced_classification_roc_curve.png', dpi=300, bbox_inches='tight')


def visualize_regression_results():
    results = pd.read_csv('../output/regression_results.csv')
    feature_importance = pd.read_csv('../output/regression_feature_importance.csv')

    plt.figure(figsize=(10, 6))
    plt.scatter(results['y_true'], results['gb_pred'], alpha=0.5)
    plt.plot([0, results['y_true'].max()], [0, results['y_true'].max()], 'r--')
    plt.xlabel('Actual Adoption Lag (days)')
    plt.ylabel('Predicted Adoption Lag (days)')
    plt.title('Actual vs Predicted Adoption Lag', fontsize=14)
    plt.savefig('../output/figures/regression_actual_vs_predicted.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    errors = results['y_true'] - results['gb_pred']
    sns.histplot(errors, kde=True)
    plt.axvline(0, color='r', linestyle='--')
    plt.xlabel('Prediction Error (days)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors for Adoption Lag', fontsize=14)
    plt.savefig('../output/figures/regression_error_distribution.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(10).sort_values('Importance')
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 10 Features for Predicting Adoption Lag', fontsize=14)
    plt.tight_layout()
    plt.savefig('../output/figures/regression_feature_importance.png', dpi=300, bbox_inches='tight')

def create_combined_policy_insights():
    model_data = pd.read_csv('../output/preprocessed/model_ready_data.csv')

    plt.figure(figsize=(12, 6))
    region_adoption = model_data.groupby('Region')['Copied_From_Neighbor'].mean().reset_index()
    region_adoption = region_adoption.sort_values('Copied_From_Neighbor', ascending=False)
    
    sns.barplot(x='Region', y='Copied_From_Neighbor', data=region_adoption)
    plt.ylabel('Proportion of Policies Copied from Neighbors')
    plt.title('Regional Differences in Policy Copying Behavior', fontsize=14)
    plt.savefig('../output/figures/regional_adoption_patterns.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=model_data[model_data['Copied_From_Neighbor'] == True],
        x='Neighbor_Adoption_Rate',
        y='Adoption_Lag',
        hue='Region',
        alpha=0.7
    )
    plt.xlabel('Neighbor Adoption Rate')
    plt.ylabel('Adoption Lag (days)')
    plt.title('Relationship Between Neighbor Adoption Rate and Adoption Lag', fontsize=14)
    plt.savefig('../output/figures/adoption_rate_vs_lag.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(14, 8))
    policy_copy_rate = model_data.groupby('MeasureDesc')['Copied_From_Neighbor'].mean().reset_index()
    policy_copy_rate = policy_copy_rate.sort_values('Copied_From_Neighbor', ascending=False)
    
    sns.barplot(x='Copied_From_Neighbor', y='MeasureDesc', data=policy_copy_rate)
    plt.xlabel('Proportion of Policies Copied from Neighbors')
    plt.title('Copying Frequency by Policy Type', fontsize=14)
    plt.tight_layout()
    plt.savefig('../output/figures/policy_type_copying.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(12, 6))
    yearly_adoption = model_data.groupby('Year')['Copied_From_Neighbor'].mean().reset_index()
    
    sns.lineplot(x='Year', y='Copied_From_Neighbor', data=yearly_adoption, marker='o')
    plt.ylabel('Proportion of Policies Copied from Neighbors')
    plt.title('Evolution of Policy Copying Over Time', fontsize=14)
    plt.savefig('../output/figures/adoption_timeline.png', dpi=300, bbox_inches='tight')
    

def create_regional_policy_comparison():
    model_data = pd.read_csv('../output/preprocessed/model_ready_data.csv')

    if 'Region' not in model_data.columns:
        region_mapping = {
            'Northeast': ['CT', 'MA', 'ME', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'],
            'South': ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'DC', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX'],
            'Midwest': ['IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI'],
            'West': ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY', 'CA']
        }
        
        model_data['Region'] = model_data['LocationAbbr'].map(
            {state: region for region, states in region_mapping.items() for state in states}
        )

    plt.figure(figsize=(12, 8))
    region_adoption = model_data.groupby('Region')['Copied_From_Neighbor'].mean().reset_index()
    region_adoption = region_adoption.sort_values('Copied_From_Neighbor', ascending=False)
    
    sns.barplot(x='Region', y='Copied_From_Neighbor', data=region_adoption)
    plt.ylabel('Proportion of Policies Copied from Neighbors')
    plt.title('Regional Differences in Policy Copying Behavior', fontsize=14)

    for i, row in enumerate(region_adoption.itertuples()):
        plt.text(i, row.Copied_From_Neighbor + 0.02, 
                f"{row.Copied_From_Neighbor:.1%}", 
                ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../output/figures/enhanced_regional_adoption_patterns.png', dpi=300, bbox_inches='tight')



def main():
    setup_visualization()
    
    print("Creating classification model visualizations...")
    visualize_classification_results()
    
    print("Creating regression model visualizations...")
    visualize_regression_results()
    
    print("Creating policy insight visualizations...")
    create_combined_policy_insights()
    
    print("Creating regional policy comparison visualizations...")
    create_regional_policy_comparison()
    
    print("Visualization complete! All figures saved to output/figures directory.")

if __name__ == "__main__":
    main()
