import numpy as np


class LinearRegression:
    #Multiple linear regression (OLS) implemented from scratch.

    def __init__(self):
        self.beta = None  # [intercept, coef1, coef2, ...]
        self.cost_history = []

    @staticmethod
    def _add_intercept(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.column_stack([np.ones(len(X)), X])

    def fit(self, X, y):
        #Closed-form solution: solves (XᵀX) β = Xᵀy.
        X = self._add_intercept(X)
        y = np.asarray(y, dtype=float)
        self.beta = np.linalg.solve(X.T @ X, X.T @ y)
        return self

    def fit_gradient_descent(self, X, y, lr=0.01, n_iter=5000):
        #Gradient descent on the mean squared error.
        #Works best with standardized features (see standardize()).
        X = self._add_intercept(X)
        y = np.asarray(y, dtype=float)
        n = len(y)
        self.beta = np.zeros(X.shape[1])
        self.cost_history = []
        for _ in range(n_iter):
            error = X @ self.beta - y
            self.beta -= lr * (2 / n) * (X.T @ error)
            self.cost_history.append(np.mean(error ** 2))
        return self

    def predict(self, X):
        if self.beta is None:
            raise ValueError("Call fit() or fit_gradient_descent() first.")
        return self._add_intercept(X) @ self.beta


def standardize(X):
    #Return standardized X (mean 0, std 1) along with its mean and std.
    X = np.asarray(X, dtype=float)
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    return (X - mu) / sigma, mu, sigma


def r2(y, y_pred):
    y, y_pred = np.asarray(y), np.asarray(y_pred)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def rmse(y, y_pred):
    return np.sqrt(np.mean((np.asarray(y) - np.asarray(y_pred)) ** 2))


def mae(y, y_pred):
    return np.mean(np.abs(np.asarray(y) - np.asarray(y_pred)))