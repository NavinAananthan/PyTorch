import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class LogisticRegression:
    def __init__(self, learning_rate, num_epochs, X, Y):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Creating a synthetic dataset
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))
        self.n_samples, self.n_features = self.X.shape

        # Initialize the model, loss, and optimizer
        self.model = LogisticRegressionModel(self.n_features)
        self.loss = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        for epoch in range(self.num_epochs):
            # Forward pass and loss
            y_pred = self.model(self.X).squeeze()
            l = self.loss(y_pred, self.Y)
            
            # Backward pass
            l.backward()
            
            # Update weights
            self.optimizer.step()
            self.optimizer.zero_grad()

            if (epoch + 1) % 10 == 0:
                print(f'epoch: {epoch + 1}, loss: {l.item():.4f}')

    def predict(self, X):
        with torch.no_grad():
            y_pred = self.model(X)
            y_pred_cls = (y_pred > 0.5).float()
        return y_pred_cls

    def plot_decision_boundary(self):
        X_numpy = self.X.numpy()
        Y_numpy = self.Y.numpy()
        x_min, x_max = X_numpy[:, 0].min() - 1, X_numpy[:, 0].max() + 1
        y_min, y_max = X_numpy[:, 1].min() - 1, X_numpy[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
        Z = self.predict(grid_tensor)
        Z = Z.view(xx.shape)
        plt.contourf(xx, yy, Z.numpy(), cmap=plt.cm.Spectral)
        plt.scatter(X_numpy[:, 0], X_numpy[:, 1], c=Y_numpy, cmap=plt.cm.Spectral)
        plt.show()


# Instantiate and train the model
X, Y = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0, random_state=42)
learning_rate = 0.01
num_epochs = 200
logistic_regression = LogisticRegression(learning_rate, num_epochs, X, Y)
logistic_regression.train()

input_value = torch.tensor([[1.0,10.0]])
predicted_probabilities = logistic_regression.predict(input_value)
predicted_class = (predicted_probabilities > 0.5).float().item()

print(predicted_class)

# Plot the decision boundary
#logistic_regression.plot_decision_boundary()
