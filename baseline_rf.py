import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Define the training data
X = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 0, 0]])
y = np.array([1, 0, 1, 0, 1, 0])

# Train the Random Forest model
model = RandomForestClassifier()
model.fit(X, y)

# Make a prediction
new_vote = np.array([1, 0, 1])
prediction = model.predict([new_vote])[0]
if prediction == 1:
    print("New vote is accepted")
else:
    print("New vote is rejected")
