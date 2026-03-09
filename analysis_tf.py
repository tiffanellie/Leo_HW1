import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure output directories exist
os.makedirs('images', exist_ok=True)
os.makedirs('models', exist_ok=True)

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
    print("Loading data splits...")
    X_train_scaled = joblib.load('models/X_train_scaled.joblib')
    X_test_scaled = joblib.load('models/X_test_scaled.joblib')
    y_train = joblib.load('models/y_train.joblib')
    y_test = joblib.load('models/y_test.joblib')
    
    print("\n--- Training Neural Network (MLP) ---")
    tf.random.set_seed(42)
    mlp = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='linear')
    ])
    mlp.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    history = mlp.fit(X_train_scaled, y_train, epochs=30, batch_size=64, validation_split=0.2, verbose=1)
    
    # Plot training history
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history['loss'], label='Train Loss (MSE)')
    ax.plot(history.history['val_loss'], label='Val Loss (MSE)')
    ax.set_title('MLP Training History')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    plt.tight_layout()
    plt.savefig('images/mlp_training_history.png')
    plt.close()
    
    mlp_pred = mlp.predict(X_test_scaled).flatten()
    mlp.save('models/MLP_model.keras')
    
    results = []
    results.append({
        'Model': 'Neural Network (MLP)',
        'MAE': mean_absolute_error(y_test, mlp_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, mlp_pred)),
        'R2': r2_score(y_test, mlp_pred)
    })
    plot_pred_vs_actual(y_test, mlp_pred, 'Neural Network (MLP)')
    
    # Save Metrics
    results_df = pd.DataFrame(results)
    results_df.to_csv('models/model_metrics_tf.csv', index=False)
    
    print("\n--- Model Evaluation Summary (TensorFlow) ---")
    print(results_df)

if __name__ == "__main__":
    main()
