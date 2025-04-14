import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_data():
    X_train = np.load('../output/preprocessed/X_train_reg.npy')
    X_test = np.load('../output/preprocessed/X_test_reg.npy')
    y_train = np.load('../output/preprocessed/y_train_reg.npy')
    y_test = np.load('../output/preprocessed/y_test_reg.npy')
    
    feature_names = pd.read_csv('../output/preprocessed/feature_names.csv', header=0)
    feature_names = feature_names.iloc[:, 0].tolist()
    
    return X_train, X_test, y_train, y_test, feature_names

def train_gradient_boosting(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best gradient boosting parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, feature_names):
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation:")
    print(f"RMSE: {rmse:.2f} days")
    print(f"MAE: {mae:.2f} days")
    print(f"R²: {r2:.4f}")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Importances:")
    for i, idx in enumerate(indices[:10]):
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    return y_pred

def main():
    os.makedirs('../models', exist_ok=True)

    X_train, X_test, y_train, y_test, feature_names = load_data()
    
    print("\nTraining Gradient Boosting model...")
    gb_model = train_gradient_boosting(X_train, y_train)

    print("\n--- Gradient Boosting Evaluation ---")
    gb_pred = evaluate_model(gb_model, X_test, y_test, feature_names)

    joblib.dump(gb_model, '../models/regression_model.pkl')

    pd.DataFrame({
        'y_true': y_test,
        'gb_pred': gb_pred
    }).to_csv('../output/regression_results.csv', index=False)
    
    print("\nRegression model trained and evaluated successfully")

if __name__ == "__main__":
    main()
