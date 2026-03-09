# King County Housing Data Science Pipeline & Prediction App

This repository constitutes the final data science project predicting King County housing prices through extensive EDA, feature tuning, grid-searched modeling, Neural Networks, and highly-interpretable SHAP Machine Learning UI deployments using Streamlit.

## Complete Project Structure:
1. `kc_house_data 2.csv` (Original dataset)
2. `analysis_trees.py` (Performs Part 1 & Part 2 GridSearched Trees)
3. `analysis_nn.py` (Trains PyTorch Neural Network MLP)
4. `app.py` (Final submission Streamlit Application mapping Parts 1, 2, 3, 4)
5. `models/` & `images/` (Persisted grid search results, metrics, and SHAP visual exports).

## How to Run Locally

### 1. Re-Generate Model Artefacts
If you intend to completely recalculate the models from scratch (Warning: 5-Fold gridsearch takes ~2-5 mins):
```bash
python3 analysis_trees.py
python3 analysis_nn.py
```
This generates `model_metrics_trees.csv`, `model_metrics_tf.csv` and all `.joblib` & `.pth` weights.

### 2. Launch the Web Application
Our `app.py` encapsulates all insights required by the rubric across highly interactive tabs:
```bash
python3 -m streamlit run app.py
```

## Hosted Cloud App
The final required deployment link for the grading rubric:
*(Link goes here for Streamlit Community Cloud after uploading these files)*.

## Included Models
* Linear Regression (Baseline)
* Decision Tree (GridSearchCV)
* Random Forest (GridSearchCV)
* LightGBM (GridSearchCV - *Best Estimator*)
* Multi-Layer Perceptron (PyTorch)
