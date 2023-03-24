import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the training data
X = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 0, 0]])
y = np.array([1, 0, 1, 0, 1, 0])

# Define the LSTM model
class PaxosLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PaxosLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        hidden = hidden[-1, :]
        out = self.activation(self.fc1(hidden))
        return out

# Define the loss function and optimizer
model = PaxosLSTM(3, 8, 1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1000):
    inputs = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    targets = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Make a prediction
new_vote = np.array([1, 0, 1])
new_vote_tensor = torch.tensor(new_vote, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
prediction = model(new_vote_tensor)
if prediction.item() >= 0.5:
    print("New vote is accepted")
else:
    print("New vote is rejected")
