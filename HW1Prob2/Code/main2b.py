import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load data
data = pd.read_csv('/home/ssiraz/Development/PhD Courses/Real time Machine Learning/Housing.csv')  # Replace with your actual file

# One-hot encoding categorical features
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
encoder = OneHotEncoder(drop='first', sparse_output=False)
categorical_encoded = encoder.fit_transform(data[categorical_cols])
categorical_encoded = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_cols))

data = data.drop(columns=categorical_cols)
data = pd.concat([data, categorical_encoded], axis=1)

# Splitting features and target
X = data.drop(columns=['price']).values
y = data['price'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize model
input_size = X_train.shape[1]
model = MLP(input_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    batch_losses = []
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    
    train_loss = np.mean(batch_losses)
    train_losses.append(train_loss)
    
    # Validation loss
    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss.append(loss.item())
    val_losses.append(np.mean(val_loss))
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_losses[-1]:.4f}')

# Plot training and validation loss
plt.figure(figsize=(10,5))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('training_validation_loss.png')  # Save the plot
plt.show()

# Evaluate the model
model.eval()
test_losses = []
y_pred = []
y_true = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        test_losses.append(criterion(outputs, batch_y).item())
        y_pred.extend(outputs.numpy().flatten())
        y_true.extend(batch_y.numpy().flatten())

final_test_mse = np.mean(test_losses)
final_test_mae = mean_absolute_error(y_true, y_pred)
final_r2 = r2_score(y_true, y_pred)

print(f'Test MSE: {final_test_mse:.2f}')
print(f'Test MAE: {final_test_mae:.2f}')
print(f'R² Score: {final_r2:.2f}')

# Save evaluation metrics to CSV
metrics_df = pd.DataFrame({'Metric': ['MSE', 'MAE', 'R² Score'], 'Value': [final_test_mse, final_test_mae, final_r2]})
metrics_df.to_csv('test_metrics.csv', index=False)

# Model complexity
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model Complexity: {total_params} trainable parameters')
