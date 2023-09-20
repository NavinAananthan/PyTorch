import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class LinearRegression:
    def __init__(self, learning_rate, num_epochs, X, Y):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Modifyingthe datasets
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))
        self.Y = self.Y.view(self.Y.shape[0], 1)
        self.n_samples, self.n_features = self.X.shape

        # Initialize the model, loss, and optimizer
        self.model = LinearRegressionModel(self.n_features, 1)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        for epoch in range(self.num_epochs):
            # Forward pass and loss
            y_pred = self.model(self.X)
            l = self.loss(y_pred, self.Y)
            
            # Backward pass
            l.backward()
            
            # Update weights
            self.optimizer.step()
            self.optimizer.zero_grad()

            if (epoch + 1) % 10 == 0:
                print(f'epoch: {epoch + 1}, loss: {l.item():.4f}')

    def predict(self, X):
        return self.model(X)

    def plot(self):
        predicted = self.predict(self.X).detach().numpy()
        plt.plot(self.X.numpy(), self.Y.numpy(), 'ro')
        plt.plot(self.X.numpy(), predicted, 'b')
        plt.show()



# Instantiate and train the model
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

learning_rate = 0.01
num_epochs = 500
linear_regression = LinearRegression(learning_rate, num_epochs, X_numpy, Y_numpy)
linear_regression.train()

# Plot the results
linear_regression.plot()
