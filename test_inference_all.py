import joblib
import pandas as pd

X = joblib.load('X_processed.joblib')

input_data = {
    'bedrooms': 3, 'bathrooms': 2.0, 'sqft_living': 1500, 'sqft_lot': 5000,
    'floors': 1.0, 'waterfront': 0, 'view': 0, 'condition': 3, 'grade': 7,
    'sqft_above': 1500, 'sqft_basement': 0, 'yr_built': 1990, 'yr_renovated': 0,
    'zipcode': 98000, 'lat': 47.5, 'long': -122.2, 'sqft_living15': 1500,
    'sqft_lot15': 5000, 'sale_year': 2014, 'sale_month': 5
}
input_df = pd.DataFrame([input_data])[X.columns]

models = {
    'Linear Regression': 'Linear_Regression_model.joblib',
    'Lasso': 'Lasso_model.joblib',
    'Ridge': 'Ridge_model.joblib',
    'CART (Decision Tree)': 'CART_Decision_Tree_model.joblib',
    'Random Forest': 'Random_Forest_model.joblib',
    'LightGBM': 'LightGBM_model.joblib'
}

for name, path in models.items():
    print(f"Testing {name}...")
    try:
        model = joblib.load(path)
        prediction = model.predict(input_df)[0]
        print(f"  Prediction: {prediction}")
    except Exception as e:
        print(f"  Error: {e}")

