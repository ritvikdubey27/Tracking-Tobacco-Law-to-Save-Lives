import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from imblearn.over_sampling import SMOTE

def load_data():
    X_train = np.load('../output/preprocessed/X_train_clf.npy')
    X_test = np.load('../output/preprocessed/X_test_clf.npy')
    y_train = np.load('../output/preprocessed/y_train_clf.npy')
    y_test = np.load('../output/preprocessed/y_test_clf.npy')
    
    feature_names = pd.read_csv('../output/preprocessed/feature_names.csv', header=0)
    feature_names = feature_names.iloc[:, 0].tolist()
    
    return X_train, X_test, y_train, y_test, feature_names

def handle_class_imbalance(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    return X_train_balanced, y_train_balanced

def train_logistic_regression(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': [None, 'balanced']
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best logistic regression parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': [None, 'balanced']
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best random forest parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, feature_names):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print("\nModel Evaluation:")
    print(f"ROC AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(report)

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nFeature Importances:")
        for i, idx in enumerate(indices[:10]):
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        indices = np.argsort(np.abs(coefficients))[::-1]
        
        print("\nFeature Coefficients:")
        for i, idx in enumerate(indices[:10]):
            print(f"{i+1}. {feature_names[idx]}: {coefficients[idx]:.4f}")
    
    return y_pred, y_prob

def main():
    os.makedirs('../models', exist_ok=True)

    X_train, X_test, y_train, y_test, feature_names = load_data()

    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train)
    
    print("Training Logistic Regression model...")
    lr_model = train_logistic_regression(X_train_balanced, y_train_balanced)
    
    print("\nTraining Random Forest model...")
    rf_model = train_random_forest(X_train_balanced, y_train_balanced)
    
    # Evaluate models
    print("\nLogistic Regression Evaluation ---")
    lr_pred, lr_prob = evaluate_model(lr_model, X_test, y_test, feature_names)
    
    print("\nRandom Forest Evaluation ---")
    rf_pred, rf_prob = evaluate_model(rf_model, X_test, y_test, feature_names)
    
    joblib.dump(rf_model, '../models/classification_model.pkl')

    pd.DataFrame({
        'y_true': y_test,
        'lr_pred': lr_pred,
        'lr_prob': lr_prob,
        'rf_pred': rf_pred,
        'rf_prob': rf_prob
    }).to_csv('../output/classification_results.csv', index=False)
    
    print("\nClassification models trained and evaluated successfully!")

if __name__ == "__main__":
    main()
