import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# --- Page Config ---
st.set_page_config(page_title="King County Housing Predictor", layout="wide")

# --- Load Artifacts ---
@st.cache_data
def load_data():
    df = pd.read_csv('kc_house_data 2.csv')
    return df

@st.cache_resource
def load_models_and_assets():
    scaler = joblib.load('models/scaler.joblib')
    X_cols = joblib.load('models/X_columns.joblib')
    X_train = joblib.load('models/X_train.joblib')
    
    models = {
        'Linear Regression': joblib.load('models/Linear_Regression_model.joblib'),
        'Decision Tree': joblib.load('models/Decision_Tree_model.joblib'),
        'Random Forest': joblib.load('models/Random_Forest_model.joblib'),
        'LightGBM': joblib.load('models/LightGBM_model.joblib')
    }
    
    metrics_trees = pd.read_csv('models/model_metrics_trees.csv')
    metrics_tf = pd.read_csv('models/model_metrics_tf.csv')
    combined_metrics = pd.concat([metrics_trees, metrics_tf], ignore_index=True)
    
    with open('models/hyperparameters_trees.json', 'r') as f:
        hyperparams = json.load(f)
        
    return scaler, X_cols, X_train, models, combined_metrics, hyperparams

st.title("King County Housing: Data Science Pipeline")

# Load Everything
try:
    df_raw = load_data()
    scaler, X_cols, X_train, models, combined_metrics, hyperparams = load_models_and_assets()
except Exception as e:
    st.error(f"Error loading models or data. Did you run the analysis scripts? Error: {e}")
    st.stop()

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary", 
    "Descriptive Analytics", 
    "Model Performance", 
    "Explainability & Inference"
])

# ==========================================
# TAB 1: EXECUTIVE SUMMARY
# ==========================================
with tab1:
    st.header("Executive Summary")
    
    st.markdown("""
    ### Project Overview & Dataset
    This project focuses on predicting residential housing prices in King County, Washington (the county comprising Seattle and its surrounding areas). The dataset contains historic home sales primarily spanning 2014 and 2015, capturing comprehensive feature sets that govern housing evaluations. **The specific prediction target is the `price` column**, representing the numeric dollar value the home sold for. We have a robust table of ~21,600 rows and 17 accompanying features, which combine numerical attributes (like square footage and number of bathrooms) alongside critical categorical characteristics (like building grade and waterfront presence).
    
    ### Why this Problem Matters
    Accurately estimating property values is a cornerstone problem in the real estate, financial, and civic planning sectors. A rigorous pricing model significantly improves property tax assessments, helps buyers or sellers make more informed market decisions, and allows developers or investors to isolate which specific factors (e.g., renovations, extensions, or architectural grade) mathematically drive the most ROI. Bridging the gap between raw data and interpretable pricing directly supports data-driven valuation in a high-stakes economic environment.
    
    ### Approach & Key Findings
    In this analysis, we fundamentally approached the challenge as a supervised regression task. We thoroughly cleaned the sales dates into components, resolved formatting issues, and scaled the numerical features. We then trained and strictly evaluated five independent algorithms ranging from parametric baselines (Linear Regression) to non-linear tree ensembles (Grid-Searched Decision Trees, Random Forests, LightGBM) and deep neural networks (PyTorch Multi-Layer Perceptron).
    
    **Key Findings:**
    - Non-linear, tree-based models (specifically LightGBM and Random Forest) overwhelmingly dominated performance over linear models due to the highly intricate socio-geographic factors impacting property values.
    - *Square footage of living space*, *overall building grade*, and *waterfront status* dynamically proved to be the top three most influential drivers pushing home prices upward.
    - Specifically, our LightGBM model was able to explain over **87% of the variance (R²)** in housing prices within the test bracket, demonstrating profound capability in reliably predicting King County market sales.
    """)

# ==========================================
# TAB 2: DESCRIPTIVE ANALYTICS
# ==========================================
with tab2:
    st.header("Exploratory Data Analysis")
    st.write("A deep dive into the distributions, densities, and governing variables that shape the housing data.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Plot 1
        st.image("images/target_distribution.png", use_container_width=True)
        st.markdown("**Target Distribution (Price):** The vast majority of standard housing prices hover under $1M, establishing a significant rightward skew. This heavily skewed distribution is standard for extreme-wealth assets, suggesting we have outliers in the multi-million dollar range that models will need to accommodate.")
        
        # Plot 3
        st.image("images/price_vs_sqft.png", use_container_width=True)
        st.markdown("**Square Footage vs. Price:** There is a highly visible, almost exponential positive linear relationship between living space and target price. The variance widens substantially as square footage eclipses traditional family home sizes, revealing luxury modifiers beyond mere space.")
        
        # Plot 5
        st.image("images/price_by_grade.png", use_container_width=True)
        st.markdown("**Impact of Building Grade:** The King County grading system assigns arbitrary construction quality scores. As shown in the violin groupings, homes granted a grade of 9 or above jump rapidly into premium brackets, indicating categorical prestige is deeply priced in.")
        
    with col2:
        # Plot 2
        st.image("images/price_by_waterfront.png", use_container_width=True)
        st.markdown("**Waterfront Status:** Though vastly outnumbered by inland properties, waterfront homes demand a staggering premium in King County. The baseline median price of waterfront estates is more than double that of typical distributions, making it a critical binary feature.")
        
        # Plot 4
        st.image("images/price_by_condition.png", use_container_width=True)
        st.markdown("**Property Condition Impact:** While 'grade' strictly defined architectural quality, 'condition' represents maintenance. Surprisingly, the leap from an average condition (3) to an excellent condition (5) produces only a moderate linear boost compared to the extreme leaps caused by grading.")
        
        # Plot 6
        st.image("images/correlation_heatmap.png", use_container_width=True)
        st.markdown("**Correlation Matrix:** Deep reds confirm tight geometric multi-collinearity among size-based features (e.g., `sqft_living`, `sqft_above`, `bathrooms`). Price correlates strongest to `sqft_living` and `grade`, identifying the overarching bedrock of our regression task.")

# ==========================================
# TAB 3: MODEL PERFORMANCE
# ==========================================
with tab3:
    st.header("Predictive Analytics & Model Evaluation")
    st.write("Cross-validated test set results (30% holdout). All continuous models aim to minimize RMSE/MAE, while maximizing pseudo-accuracy (R² variance explained).")
    
    # 1. Metrics Table
    st.dataframe(combined_metrics.style.highlight_max(subset=['R2'], color='lightgreen').highlight_min(subset=['RMSE'], color='lightgreen'), use_container_width=True)
    
    # 2. Bar Chart of Key Metric
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='RMSE', y='Model', data=combined_metrics.sort_values('RMSE'), palette='viridis', ax=ax)
    ax.set_title("Test-Set RMSE Across All Models (Lower is Better)")
    st.pyplot(fig)
    
    st.markdown("""
    **Model Performance Summary:**
    The **LightGBM** model performed overwhelmingly the best, achieving the lowest Error rate (RMSE ~ 131k) and highest variance captured (R² ~ 0.87). I was surprised that the Keras/PyTorch Multi-Layer Perceptron neural network, while robust, consistently underperformed compared to grid-searched boosted trees; this is a classic hallmark of tabular, structured data where spatial relationships favour ensemble gradient boosters over dense networks. The trade-offs are also visible: linear regressions train instantaneously but fail to capture geographic depth, while trees map the non-linear boundaries flawlessly at the modest expense of slight overfitting risks and longer train times.
    """)
    
    # 3. Hyperparameters
    st.subheader("Optimal Hyperparameters (from GridSearchCV)")
    st.json(hyperparams)
    
    # 4. Actual vs Predicted Plots
    st.subheader("Predicted vs Actual Scatter Plots")
    colA, colB, colC = st.columns(3)
    
    with colA:
        st.image("images/Linear_Regression_scatter.png", use_container_width=True)
        st.image("images/Random_Forest_scatter.png", use_container_width=True)
    with colB:
        st.image("images/Decision_Tree_scatter.png", use_container_width=True)
        st.image("images/LightGBM_scatter.png", use_container_width=True)
    with colC:
        st.image("images/Neural_Network_(MLP)_scatter.png", use_container_width=True)


# ==========================================
# TAB 4: EXPLAINABILITY & INFERENCE
# ==========================================
def st_shap(plot, height=None):
    from streamlit.components.v1 import html
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    html(shap_html, height=height)

with tab4:
    st.header("Explainability & Interactive Inference (SHAP)")
    
    st.markdown("### Global Feature Impact (LightGBM)")
    st.write("How our highest-accuracy model structurally assigns importance across the 17 variables. We use SHAP to unpack the internal ensemble routing mechanism.")
    
    col_shap1, col_shap2 = st.columns(2)
    with col_shap1:
        st.image("images/shap_summary.png", caption="SHAP Summary (Beeswarm) Plot: Density and vector direction of features.")
        st.markdown("""
        **Interpretation (Beeswarm):**
        - **Directionality:** Red dots (high values) on `grade` and `waterfront` heavily push the SHAP value rightward (positive impact on price). This visually proves that high architectural grades and luxury placement reliably inflate home prices. 
        - **Magnitude:** High values of `sqft_living` have enormous standard deviation spread on the right side, meaning big homes contribute astronomically to final predictions.
        """)
    with col_shap2:
        st.image("images/shap_bar.png", caption="Absolute Mean SHAP values: Ranking raw feature importance.")
        st.markdown("""
        **Interpretation (Bar):**
        - **Top Movers:** The model unequivocally relies on `sqft_living` and `grade` as the primary mathematical levers. 
        - **Domain Strategy:** For decision-makers and developers, simply acquiring larger lateral land is insufficient; the architectural 'grade' assigned by the county is statistically more impactful than the raw 'condition', meaning luxury aesthetics return massive ROI compared to basic infrastructural maintenance upgrades.
        """)

    st.markdown("---")
    st.subheader("Interactive Pricing Engine")
    st.write("Use the sliders to fabricate a custom housing profile. The resulting SHAP waterfall tracks exactly why the specific home is priced the way it is.")
    
    # Define an average baseline property
    default_vals = df_raw.median(numeric_only=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        user_sqft = st.number_input("Square Feet (Living)", min_value=500, max_value=10000, value=int(default_vals.get('sqft_living', 1910)))
        user_beds = st.slider("Bedrooms", 1, 10, int(default_vals.get('bedrooms', 3)))
        user_baths = st.slider("Bathrooms", 1.0, 8.0, float(default_vals.get('bathrooms', 2.25)), step=0.25)
    with c2:
        user_grade = st.slider("King County Grade", 1, 13, int(default_vals.get('grade', 7)))
        user_condition = st.slider("Condition (1-5)", 1, 5, int(default_vals.get('condition', 3)))
    with c3:
        user_yr = st.number_input("Year Built", 1900, 2015, int(default_vals.get('yr_built', 1975)))
        user_water = st.selectbox("Waterfront?", [0, 1])

    # Build Inference Dictionary using averages for hidden complex variables
    inference_dict = {
        'bedrooms': user_beds,
        'bathrooms': user_baths,
        'sqft_living': user_sqft,
        'sqft_lot': default_vals['sqft_lot'],
        'floors': default_vals['floors'],
        'waterfront': user_water,
        'view': default_vals['view'],
        'condition': user_condition,
        'grade': user_grade,
        'sqft_above': default_vals['sqft_above'],
        'sqft_basement': default_vals['sqft_basement'],
        'yr_built': user_yr,
        'yr_renovated': default_vals['yr_renovated'],
        'yr_renovated': default_vals['yr_renovated'],
        'sqft_living15': default_vals['sqft_living15'],
        'sqft_lot15': default_vals['sqft_lot15'],
        'sale_year': 2015,
        'sale_month': 6
    }
    
    input_df = pd.DataFrame([inference_dict])[X_cols]
    scaled_input = scaler.transform(input_df)
    
    selected_model_name = st.selectbox("Execution Engine:", list(models.keys()), index=3)
    active_model = models[selected_model_name]
    
    pred_val = active_model.predict(scaled_input)[0]
    
    st.markdown(f"### 💰 Estimated Value: **${pred_val:,.2f}**")
    
    # Generate local SHAP Waterfall
    if selected_model_name == 'LightGBM':
        st.write("### Local Prediction Mechanics (Waterfall)")
        # Calculate single SHAP prediction context
        explainer = shap.TreeExplainer(active_model)
        shap_vals = explainer(scaled_input)
        
        # We replace the scaled feature names with real values for readability in the plot
        shap_vals.data = input_df.values
        shap_vals.feature_names = list(input_df.columns)
        
        fig_water, ax_water = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_vals[0], show=False)
        plt.tight_layout()
        st.pyplot(fig_water)
        
        st.markdown("**Explanation of User Input Waterfall:** The baseline starts at the overall average housing price expected by the LightGBM algorithm. Every red bar identifies how your specific configurations (e.g. moving the location or upgrading square-footage) drove the predicted property value massively upward, while blue arrows pushed it downward.")
    else:
        st.info("Dynamic Waterfall SHAP explanations are exclusively pre-configured for the primary LightGBM Model.")
