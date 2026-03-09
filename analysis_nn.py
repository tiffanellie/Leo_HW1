import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x)

def main():
    print("Loading data splits...")
    X_train_scaled = joblib.load('models/X_train_scaled.joblib')
    X_test_scaled = joblib.load('models/X_test_scaled.joblib')
    y_train = joblib.load('models/y_train.joblib')
    y_test = joblib.load('models/y_test.joblib')
    
    # We need to split a validation set out of the training data
    from sklearn.model_selection import train_test_split
    X_t, X_v, y_t, y_v = train_test_split(X_train_scaled, y_train.values, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_t_tensor = torch.tensor(X_t, dtype=torch.float32)
    y_t_tensor = torch.tensor(y_t, dtype=torch.float32).view(-1, 1)
    
    X_v_tensor = torch.tensor(X_v, dtype=torch.float32)
    y_v_tensor = torch.tensor(y_v, dtype=torch.float32).view(-1, 1)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    train_dataset = TensorDataset(X_t_tensor, y_t_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    print("\n--- Training Neural Network (PyTorch MLP) ---")
    torch.manual_seed(42)
    
    model = MLPModel(X_train_scaled.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 30
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            v_preds = model(X_v_tensor)
            v_loss = criterion(v_preds, y_v_tensor).item()
            val_losses.append(v_loss)
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss (MSE): {train_loss:.2f} | Val Loss: {v_loss:.2f}")
    
    # Plot training history
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='Train Loss (MSE)')
    ax.plot(val_losses, label='Val Loss (MSE)')
    ax.set_title('MLP Training History')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    plt.tight_layout()
    plt.savefig('images/mlp_training_history.png')
    plt.close()
    
    # Predict on test set
    model.eval()
    with torch.no_grad():
        mlp_pred = model(X_test_tensor).numpy().flatten()
        
    torch.save(model.state_dict(), 'models/MLP_model.pth')
    
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
    
    print("\n--- Model Evaluation Summary (PyTorch) ---")
    print(results_df)

if __name__ == "__main__":
    main()
