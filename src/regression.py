import torch
from itertools import combinations
import psutil

def _ensure_x(func):
    def wrapper(self, X, *args, **kwargs):
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.dtype != torch.float:
            X = X.float()
        return func(self, X, *args, **kwargs)
    return wrapper

def _ensure_y(func):
    def wrapper(self, X, y, *args, **kwargs):
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y, dtype=torch.float)
        if y.ndim != 1:
            if y.ndim > 2:
                raise ValueError(f"Y must be a 1D or 2D tensor, received {y.ndim}D")
            if y.shape[1] != 1:
                raise ValueError(f"Y must be a 1D tensor, received {y.shape[1]} columns")
            y = y.flatten()
        if y.dtype != torch.float:
            y = y.float()
        return func(self, X, y, *args, **kwargs)
    return wrapper


class LinearRegression:
    @_ensure_y
    @_ensure_x
    def fit(self, X, y):
        X_aug = torch.cat(
            [
                torch.ones(X.shape[0], 1).to(X.device), 
                X
            ], 
            dim=1
        ).float()

        theta = torch.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y.float()
        theta = theta.squeeze()
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self

    
    @_ensure_x
    def predict(self, X):
        return self.intercept_ + X @ self.coef_

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)


class PairwiseLinearRegression:
    def __init__(self, memory_threshold=0.75):
        self._validate_memory_threshold(memory_threshold)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _validate_memory_threshold(self, memory_threshold):
        memory_threshold = memory_threshold or 0.75

        if not isinstance(memory_threshold, (int, float)):
            raise ValueError(f"Memory threshold must be a number, received {type(memory_threshold)}")
        
        if memory_threshold <= 0 or memory_threshold > 1:
            raise ValueError(f"Memory threshold must be between 0 and 1, received {memory_threshold}")

        self.memory_threshold = memory_threshold

    @staticmethod
    def _incremental_gram_matrix(X, y):
        G = X.T @ X
        g = X.T @ y
        sx = X.sum(dim=0)
        sy = y.sum()
        return G, g, sx, sy

    @_ensure_y
    @_ensure_x
    def fit(self, X, y):
        X = X.to(self.device)
        y = y.to(self.device)

        n, p = X.shape
        
        G, g, sx, sy = PairwiseLinearRegression._incremental_gram_matrix(X, y)

        pairs = torch.tensor(list(combinations(range(p), 2)), device=X.device)

        i = pairs[:, 0]
        j = pairs[:, 1]
        C = len(pairs)

        A = torch.zeros((C, 3, 3), dtype=torch.float, device=X.device)
        A[:, 0, 0] = n
        A[:, 0, 1] = sx[i]
        A[:, 0, 2] = sx[j]

        A[:, 1, 0] = sx[i]
        A[:, 1, 1] = G[i, i]
        A[:, 1, 2] = G[i, j]

        A[:, 2, 0] = sx[j]
        A[:, 2, 1] = G[j, i]
        A[:, 2, 2] = G[j, j]

        b = torch.zeros((C, 3), dtype=torch.float, device=X.device)
        b[:, 0] = sy
        b[:, 1] = g[i]
        b[:, 2] = g[j]

        betas, *_ = torch.linalg.lstsq(A, b)

        self.coef_ = betas[:, [1, 2]].float()
        self.intercept_ = betas[:, 0].float()
        self.i = i
        self.j = j
        self.C = C

        return self

    def _calculate_batch_size(self, element_size):
        if self.device.type == "cpu":
            free = psutil.virtual_memory().available 
        else:
            free, _ = torch.cuda.mem_get_info()
        max_bytes = free * self.memory_threshold
        return max(1, int(max_bytes // (3 * self.C * element_size)))

    @_ensure_x
    def predict(self, X):
        with torch.no_grad():
            preds = torch.empty((X.shape[0], self.C), dtype=X.dtype)
            element_size = X.element_size()
            start = 0
            while start < X.shape[0]:
                Xb = X[
                    start:
                    start + (batch_size := self._calculate_batch_size(element_size))
                ].to(self.device)

                out = (
                    self.intercept_[None, :] +
                    Xb[:, self.i] * self.coef_[:, 0] + 
                    Xb[:, self.j] * self.coef_[:, 1]
                )

                preds[start:start + batch_size] = out.cpu()
                start += batch_size
                
                del Xb, out
                torch.cuda.empty_cache()

            print(preds.shape, X.shape)
            return preds

    @_ensure_y
    @_ensure_x
    def evaluate(self, X, y, metrics=["rmse"]):
        res = {}
        with torch.no_grad():
            y_pred = self.predict(X)
            for metric in metrics:
                if metric == "rmse":
                    res[metric] = torch.sqrt(((y_pred - y[:, None])**2).mean(dim=0))
                elif metric == "mae":
                    res[metric] = (y_pred - y[:, None]).abs().mean(dim=0)
                elif metric == "r2":
                    y_mean = y.mean(dim=0)
                    ss_tot = ((y - y_mean[None, :])**2).sum(dim=0)
                    ss_res = ((y_pred - y[:, None])**2).sum(dim=0)
                    res[metric] = 1 - ss_res / ss_tot
                elif metric == "mape":
                    res[metric] = ((y_pred - y[:, None]).abs() / y[:, None]).mean(dim=0)
                else:
                    raise ValueError(f"Invalid metric: {metric}")
        return res