import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class LogisticRegressionModel(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

class LogisticRegression:
    def __init__(self, num_epochs=100, learning_rate=0.01):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scaler = None

    def preprocess_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        X_train = torch.from_numpy(X_train.astype(np.float32))
        X_test = torch.from_numpy(X_test.astype(np.float32))
        y_train = torch.from_numpy(y_train.astype(np.float32))
        y_test = torch.from_numpy(y_test.astype(np.float32))

        y_train = y_train.view(y_train.shape[0], 1)
        y_test = y_test.view(y_test.shape[0], 1)

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.model = LogisticRegressionModel(n_features)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            y_pred = self.model(X_train)
            loss = self.criterion(y_pred, y_train)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}')

    def evaluate(self, X_test, y_test):
        with torch.no_grad():
            y_predicted = self.model(X_test)
            y_predicted_cls = y_predicted.round()
            acc = y_predicted_cls.eq(y_test).sum().item() / float(y_test.shape[0])
            return acc

if __name__ == '__main__':
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    lr_model = LogisticRegression(num_epochs=100, learning_rate=0.01)
    X_train, X_test, y_train, y_test = lr_model.preprocess_data(X, y)

    lr_model.train(X_train, y_train)

    accuracy = lr_model.evaluate(X_test, y_test)
    print(f'Accuracy: {accuracy:.4f}')
