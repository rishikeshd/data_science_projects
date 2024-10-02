import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RegressorFromScratch(BaseEstimator, TransformerMixin):

    def __init__(self, learning_rate: int, n_iter: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def initialize_parameters(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    @staticmethod
    def calculate_gradients(X, y, y_pred, n_samples):
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)

        return dw, db

    def fit(self, X, y):
        # We first initialize the weights and biases for linear regression.
        # Linear regression fits the line to the points using this equation of y= (W*X) + b
        n_samples, n_features = X.shape
        self.initialize_parameters(n_features)
        cost_list = []
        iter_list = []

        for i in range(0, self.n_iter):
            # We fit the line using initialized weights and biases
            y_pred = self.feed_forward(X)

            # calculate current error
            cost = self.compute_cost(n_samples, y, y_pred)

            if i % 100 == 0:
                cost_list.append(cost)
                iter_list.append(i)
                # print(f'Cost for iteration number {i} is {cost}')

            # calculate gradients
            dw, db = self.calculate_gradients(X, y, y_pred, n_samples)

            # update weights
            self.weights = self.weights - (self.learning_rate * dw)
            self.bias = self.bias - (self.learning_rate * db)

        return cost_list, iter_list

    def feed_forward(self, X):
        return NotImplementedError("Subclass must implement feed_forward method")

    def compute_cost(self, n_samples, y, y_pred):
        return NotImplementedError("Subclass must implement compute_cost method")

    def predict(self, X):
        return NotImplementedError("Subclass must implement predict method")


class LinearRegression(RegressorFromScratch):

    def __init__(self, learning_rate: float, n_iter: int = 1000) -> None:
        super().__init__(learning_rate, n_iter)

    def feed_forward(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

    @staticmethod
    def compute_cost(n_samples, y, y_pred):
        cost = (1 / (2 * n_samples)) * np.sum(np.square(y_pred - y))
        return cost

    def predict(self, X):
        y_pred = self.feed_forward(X)
        return y_pred


class LogisticRegression(RegressorFromScratch):

    def __init__(self, learning_rate: int, n_iter: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def feed_forward(self, X):
        z = np.dot(X, self.weights)
        A = 1 / (1 + np.exp(-z))
        return A

    @staticmethod
    def compute_cost(n_samples, y, y_pred):
        epsilon = 0 # 1e-9
        cost0 = y * np.log(y_pred + epsilon)
        cost1 = (1 - y) * np.log(1 - (y_pred + epsilon))
        cost = (-1 / n_samples) * np.sum((cost0 + cost1))
        return cost

    def predict(self, X):
        threshold = 0.5
        y_pred = self.feed_forward(X)
        y_pred = np.where(y_pred >= threshold, 1, 0)  # 0.5 Threshold
        return y_pred
