import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the training data
X = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 0, 0]])
y = np.array([1, 0, 1, 0, 1, 0])

# Define the model
class PaxosModel(nn.Module):
    def __init__(self):
        super(PaxosModel, self).__init__()
        self.layer_1 = nn.Linear(3, 4)
        self.layer_2 = nn.Linear(4, 2)
        self.layer_3 = nn.Linear(2, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.layer_1(x))
        x = self.activation(self.layer_2(x))
        x = self.activation(self.layer_3(x))
        return x

# Define the loss function and optimizer
model = PaxosModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1000):
    inputs = torch.tensor(X, dtype=torch.float32)
    targets = torch.tensor(y, dtype=torch.float32)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets.unsqueeze(1))
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Make a prediction
new_vote = np.array([1, 0, 1])
new_vote_tensor = torch.tensor(new_vote, dtype=torch.float32).unsqueeze(0)
prediction = model(new_vote_tensor)
if prediction.item() >= 0.5:
    print("New vote is accepted")
else:
    print("New vote is rejected")
