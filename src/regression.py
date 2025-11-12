import torch

def _ensure_x(func):
    def wrapper(self, X, *args, **kwargs):
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.dtype != torch.float32:
            X = X.float()
        return func(self, X, *args, **kwargs)
    return wrapper

def _ensure_y(func):
    def wrapper(self, X, y, *args, **kwargs):
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
        return func(self, X, y, *args, **kwargs)
    return wrapper

class LinearRegression():
    @_ensure_y
    @_ensure_x
    def fit(self, X, y):
        X_aug = torch.cat(
            [
                torch.ones(X.shape[0], 1), 
                X
            ], 
            dim=1
        )

        theta = torch.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
        theta = theta.squeeze()

        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self

    @_ensure_x
    def predict(self, X):
        return X @ self.coef_ + self.intercept_

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)