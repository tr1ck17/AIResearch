import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
np.random.seed(42)

# data prep
df = pd.read_csv('pima.csv', header=None)
X = df.iloc[:, :8].values
y = df.iloc[:, 8].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t = torch.tensor(y_val, dtype=torch.long)

# the Network
class VanillaMLP(nn.Module):
    def __init__(self, n_inputs=8, n_hidden=16, n_outputs=2):
        super(VanillaMLP, self).__init__()
        # replacing he_init and forward
        self.layer1 = nn.Linear(n_inputs, n_hidden)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(n_hidden, n_outputs)

    def forward(self, x):
        Z1 = self.layer1(x)
        A1 = self.relu(Z1)
        Z2 = self.layer2(A1)
        # softmax handled by CrossEntropyLoss layer
        return Z2
    
# initialize model and components
model = VanillaMLP()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

# replaces accuracy func
def calculate_accuracy(logits, y_true):
    preds = torch.argmax(logits, dim=1)
    return (preds == y_true).float().mean().item()

# training loop
epochs = 1000
for epoch in range(epochs):

    model.train()

    logits = model(X_train_t)

    loss = criterion(logits, y_train_t)

    # backward_pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            train_acc = calculate_accuracy(logits, y_train_t)

            val_logits = model(X_val_t)
            val_acc = calculate_accuracy(val_logits, y_val_t)

            print(f"Epoch {epoch:04d} | Loss: {loss:.4f} | Training Accuracy: {train_acc:.4f} | Validation Accuracy: {val_acc:.4f}")