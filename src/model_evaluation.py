import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (confusion_matrix, classification_report, 
                            roc_curve, roc_auc_score, precision_recall_curve)

def evaluate_classification_model():
    results = pd.read_csv('../output/classification_results.csv')

    y_true = results['y_true']
    rf_pred = results['rf_pred']
    rf_prob = results['rf_prob']

    cm = confusion_matrix(y_true, rf_pred)

    fpr, tpr, thresholds = roc_curve(y_true, rf_prob)
    auc = roc_auc_score(y_true, rf_prob)

    report = {
        'Overall Accuracy': np.mean(rf_pred == y_true),
        'True Positive Rate': cm[1, 1] / (cm[1, 1] + cm[1, 0]),
        'True Negative Rate': cm[0, 0] / (cm[0, 0] + cm[0, 1]),
        'False Positive Rate': cm[0, 1] / (cm[0, 1] + cm[0, 0]),
        'False Negative Rate': cm[1, 0] / (cm[1, 0] + cm[1, 1]),
        'AUC': auc
    }
    
    print("\n=== Classification Model Evaluation ===")
    for metric, value in report.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, rf_pred))

    model = joblib.load('../models/classification_model.pkl')
    feature_names = pd.read_csv('../output/preprocessed/feature_names.csv', header=0)
    feature_names = feature_names.iloc[:, 0].tolist()

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)

        feature_importance.to_csv('../output/classification_feature_importance.csv', index=False)
        
        print("\nTop 10 Features for Predicting Policy Copying:")
        print(feature_importance.head(10).to_string(index=False))

    pd.DataFrame([report]).to_csv('../output/classification_metrics.csv', index=False)
    
    return report

def evaluate_regression_model():
    results = pd.read_csv('../output/regression_results.csv')

    y_true = results['y_true']
    gb_pred = results['gb_pred']

    mse = np.mean((y_true - gb_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - gb_pred))
    r2 = 1 - (np.sum((y_true - gb_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

    errors = np.abs(y_true - gb_pred)
    percentile_50 = np.percentile(errors, 50)
    percentile_75 = np.percentile(errors, 75)
    percentile_90 = np.percentile(errors, 90)

    report = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Median Error': percentile_50,
        '75th Percentile Error': percentile_75,
        '90th Percentile Error': percentile_90
    }
    
    print("\n=== Regression Model Evaluation ===")
    for metric, value in report.items():
        print(f"{metric}: {value:.2f}")

    model_data = pd.read_csv('../output/preprocessed/model_ready_data.csv')
    model_data = model_data[model_data['Copied_From_Neighbor'] == True].reset_index(drop=True)

    model = joblib.load('../models/regression_model.pkl')
    feature_names = pd.read_csv('../output/preprocessed/feature_names.csv', header=0)
    feature_names = feature_names.iloc[:, 0].tolist()

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)

        feature_importance.to_csv('../output/regression_feature_importance.csv', index=False)
        
        print("\nTop 10 Features for Predicting Adoption Lag:")
        print(feature_importance.head(10).to_string(index=False))

    pd.DataFrame([report]).to_csv('../output/regression_metrics.csv', index=False)
    
    return report

def main():
    print("Evaluating classification model...")
    classification_metrics = evaluate_classification_model()
    
    print("\nEvaluating regression model...")
    regression_metrics = evaluate_regression_model()
    
    print("\nEvaluation complete! Detailed metrics saved")

if __name__ == "__main__":
    main()
