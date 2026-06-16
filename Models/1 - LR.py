import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def linear(x,w,b):
    return w * x + b

w,b = 2,1
noise_level = 0.1
n_sample = 1000

# Generate X axis and corresponding value on the Y axis
X = np.linspace(0, 1, n_sample)
Y = [linear(x, w, b) for x in X]
Y_noise = np.array([y + random.gauss(0, noise_level) for y in Y])

X = X[:, np.newaxis]
Y_noise = Y_noise[:, np.newaxis]

plt.figure()
plt.plot(X, Y, color="r")
plt.scatter(X, Y_noise)
plt.show()

# Define the model parameters
input_size = 1
output_size = 1
learning_rate = 0.001
learning_rates = [0.5, 0.1, 0.01, 0.001, 0.0001]


model = nn.Linear(input_size, output_size) # Create a linear layer
criterion = nn.MSELoss() # Loss function
optimizer = optim.SGD(model.parameters(), lr=learning_rate) # Gradient

# Define tensors
x_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(Y_noise, dtype=torch.float32)

weights_history = []
loss_history = []
n_epochs = 1000

for epoch in range(n_epochs):
    weight = model.weight.data.clone().numpy()
    weights_history.append(weight)
    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss_history.append(float(loss))
    # Update the model (backward propagation)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

weights_history = np.array(weights_history).squeeze()
loss_history = np.array(loss_history).squeeze()

plt.figure()
plt.scatter(weights_history, loss_history, label="lr"+str(learning_rate))
plt.xlabel('Weight')
plt.ylabel('Loss')
plt.legend()
plt.show()