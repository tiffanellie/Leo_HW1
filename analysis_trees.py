import os
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# Ensure output directories exist
os.makedirs('images', exist_ok=True)
os.makedirs('models', exist_ok=True)

def load_and_preprocess_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    return df

def generate_eda_plots(df):
    print("Generating EDA plots...")
    
    # 1. Target Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['price'], bins=50, kde=True, ax=ax)
    ax.set_title('Distribution of Home Prices')
    ax.set_xlabel('Price')
    ax.set_xlim(0, 3e6) # Focusing on the bulk of the data
    plt.tight_layout()
    plt.savefig('images/target_distribution.png')
    plt.close('all')
    
    # 2. Boxplot by Waterfront
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='waterfront', y='price', data=df, ax=ax)
    ax.set_title('Price by Waterfront Status')
    ax.set_ylim(0, 4e6)
    plt.tight_layout()
    plt.savefig('images/price_by_waterfront.png')
    plt.close('all')
    
    # 3. Scatter plot Sqft vs Price
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x='sqft_living', y='price', data=df, alpha=0.3, ax=ax)
    ax.set_title('Price vs. Square Feet of Living Space')
    plt.tight_layout()
    plt.savefig('images/price_vs_sqft.png')
    plt.close('all')
    
    # 4. Bar plot of avg price by condition
    fig, ax = plt.subplots(figsize=(8, 5))
    df_cond = df.groupby('condition')['price'].mean().reset_index()
    sns.barplot(x='condition', y='price', data=df_cond, ax=ax, palette='viridis')
    ax.set_title('Average Price by House Condition')
    plt.tight_layout()
    plt.savefig('images/price_by_condition.png')
    plt.close('all')
    
    # 5. Violin plot by grade
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x='grade', y='price', data=df[df['grade'] >= 5], ax=ax)
    ax.set_title('Price Distribution by Grade (>= 5)')
    ax.set_ylim(0, 3.5e6)
    plt.tight_layout()
    plt.savefig('images/price_by_grade.png')
    plt.close('all')

    # 6. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.drop(['id', 'date', 'lat', 'long', 'zipcode'], axis=1, errors='ignore').corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('images/correlation_heatmap.png')
    plt.close('all')

def plot_pred_vs_actual(y_test, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.3)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    ax.set_title(f'{model_name}: Predicted vs Actual')
    plt.tight_layout()
    plt.savefig(f'images/{model_name.replace(" ", "_")}_scatter.png')
    plt.close('all')

def main():
    filepath = 'kc_house_data 2.csv'
    df = load_and_preprocess_data(filepath)
    
    generate_eda_plots(df)
    
    # Data Preparation
    df_proc = df.copy()
    df_proc = df_proc.drop('id', axis=1)
    df_proc['sale_year'] = df_proc['date'].str[:4].astype(int)
    df_proc['sale_month'] = df_proc['date'].str[4:6].astype(int)
    df_proc = df_proc.drop('date', axis=1)
    
    if df_proc['floors'].dtype == object:
        df_proc['floors'] = df_proc['floors'].astype(float)
        
    y = df_proc['price']
    X = df_proc.drop(['price', 'lat', 'long', 'zipcode'], axis=1, errors='ignore')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(list(X.columns), 'models/X_columns.joblib')
    # Save training data for SHAP app side
    joblib.dump(X_train, 'models/X_train.joblib')
    # Save test sets for neural network script
    joblib.dump(X_train_scaled, 'models/X_train_scaled.joblib')
    joblib.dump(X_test_scaled, 'models/X_test_scaled.joblib')
    joblib.dump(y_train, 'models/y_train.joblib')
    joblib.dump(y_test, 'models/y_test.joblib')
    
    results = []
    best_params_dict = {}

    # 1. Linear Regression (Baseline)
    print("\n--- Training Linear Regression ---")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    joblib.dump(lr, 'models/Linear_Regression_model.joblib')
    
    lr_metrics = {
        'Model': 'Linear Regression',
        'MAE': mean_absolute_error(y_test, lr_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
        'R2': r2_score(y_test, lr_pred)
    }
    results.append(lr_metrics)
    plot_pred_vs_actual(y_test, lr_pred, 'Linear Regression')
    
    # 2. Decision Tree
    print("\n--- Training Decision Tree ---")
    dt = DecisionTreeRegressor(random_state=42)
    dt_grid = {
        'max_depth': [5, 10, 15],
        'min_samples_leaf': [5, 20, 50]
    }
    gs_dt = GridSearchCV(dt, dt_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=None)
    gs_dt.fit(X_train_scaled, y_train)
    
    dt_best = gs_dt.best_estimator_
    dt_pred = dt_best.predict(X_test_scaled)
    joblib.dump(dt_best, 'models/Decision_Tree_model.joblib')
    best_params_dict['Decision Tree'] = gs_dt.best_params_
    
    results.append({
        'Model': 'Decision Tree',
        'MAE': mean_absolute_error(y_test, dt_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, dt_pred)),
        'R2': r2_score(y_test, dt_pred)
    })
    plot_pred_vs_actual(y_test, dt_pred, 'Decision Tree')
    
    # 3. Random Forest
    print("\n--- Training Random Forest ---")
    rf = RandomForestRegressor(random_state=42)
    rf_grid = {
        'n_estimators': [50, 100],
        'max_depth': [8, 12]
    }
    gs_rf = GridSearchCV(rf, rf_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=None)
    gs_rf.fit(X_train_scaled, y_train)
    
    rf_best = gs_rf.best_estimator_
    rf_pred = rf_best.predict(X_test_scaled)
    joblib.dump(rf_best, 'models/Random_Forest_model.joblib')
    best_params_dict['Random Forest'] = gs_rf.best_params_
    
    results.append({
        'Model': 'Random Forest',
        'MAE': mean_absolute_error(y_test, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'R2': r2_score(y_test, rf_pred)
    })
    plot_pred_vs_actual(y_test, rf_pred, 'Random Forest')

    # 4. LightGBM
    print("\n--- Training LightGBM ---")
    lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=None)
    lgb_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 8],
        'learning_rate': [0.05, 0.1]
    }
    gs_lgb = GridSearchCV(lgb_model, lgb_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=None)
    gs_lgb.fit(X_train_scaled, y_train)
    
    lgb_best = gs_lgb.best_estimator_
    lgb_pred = lgb_best.predict(X_test_scaled)
    joblib.dump(lgb_best, 'models/LightGBM_model.joblib')
    best_params_dict['LightGBM'] = gs_lgb.best_params_
    
    results.append({
        'Model': 'LightGBM',
        'MAE': mean_absolute_error(y_test, lgb_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, lgb_pred)),
        'R2': r2_score(y_test, lgb_pred)
    })
    plot_pred_vs_actual(y_test, lgb_pred, 'LightGBM')
    
    # Save Metrics & Params
    results_df = pd.DataFrame(results)
    results_df.to_csv('models/model_metrics_trees.csv', index=False)
    
    with open('models/hyperparameters_trees.json', 'w') as f:
        json.dump(best_params_dict, f, indent=4)
        
    print("\n--- Model Evaluation Summary (Trees) ---")
    print(results_df)
    
    # 6. SHAP Analysis
    print("\n--- Generating SHAP values for LightGBM ---")
    # Using background sample for speed
    X_sample = X_train.sample(500, random_state=42)
    X_sample_scaled = scaler.transform(X_sample)
    
    explainer = shap.TreeExplainer(lgb_best)
    shap_values = explainer.shap_values(X_sample_scaled)
    
    # SHAP Summary Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, features=X_sample, feature_names=X.columns, show=False)
    plt.tight_layout()
    plt.savefig('images/shap_summary.png')
    plt.close()

    # SHAP Bar Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, features=X_sample, feature_names=X.columns, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('images/shap_bar.png')
    plt.close()
    
    print("\nAnalysis Trees complete! Assets saved in /images and /models.")

if __name__ == "__main__":
    main()
