import torch
from itertools import combinations
import psutil

def ensure_x(func):
    def wrapper(self, X, *args, **kwargs):
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.dtype != torch.float:
            X = X.float()
        return func(self, X, *args, **kwargs)
    return wrapper

def ensure_y(func):
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
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @ensure_y
    @ensure_x
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

    
    @ensure_x
    def predict(self, X):
        return self.intercept_ + X.to(self.device) @ self.coef_.to(self.device)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)


class PairwiseLinearRegression:
    def __init__(self, metrics, memory_threshold=0.75):
        self._validate_memory_threshold(memory_threshold)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = metrics if isinstance(metrics, list) else [metrics]

    def _validate_memory_threshold(self, memory_threshold):
        memory_threshold = memory_threshold or 0.75

        if not isinstance(memory_threshold, (int, float)):
            raise ValueError(f"Memory threshold must be a number, received {type(memory_threshold)}")
        
        if memory_threshold <= 0 or memory_threshold > 1:
            raise ValueError(f"Memory threshold must be between 0 and 1, received {memory_threshold}")

        self.memory_threshold = memory_threshold

    def _calculate_batch_size(self, element_size, dim):
        if self.device.type == "cpu":
            free = psutil.virtual_memory().available 
        else:
            free, _ = torch.cuda.mem_get_info()
        max_bytes = free * self.memory_threshold
        return max(1, int(max_bytes // (3 * dim * element_size)))

    def _incremental_gram_matrix(self, X, y):
        n, p = X.shape

        G = torch.zeros((p, p), dtype=torch.float, device=self.device)
        g = torch.zeros((p,), dtype=torch.float, device=self.device)
        sx = torch.zeros((p,), dtype=torch.float, device=self.device)
        sy = torch.zeros((1,), dtype=torch.float, device=self.device)

        start = 0
        element_size = X.element_size()
        while start < n:
            # FIXME if no gpu -> no need to move to device, its already on ram. Use a view to the slice or something.
            batch_size = self._calculate_batch_size(element_size=element_size, dim=p)
            Xb = X[start:start + batch_size].cuda(self.device, non_blocking=True)
            yb = y[start:start + batch_size].cuda(self.device, non_blocking=True)

            G += Xb.T @ Xb
            g += Xb.T @ yb
            sx += Xb.sum(0)
            sy += yb.sum()

            torch.cuda.empty_cache()

            start += batch_size

        return G, g, sx, sy
        
    @ensure_y
    @ensure_x
    def fit(self, X, y):
        n, p = X.shape

        pairs = torch.tensor(list(combinations(range(p), 2)), device=X.device)

        i = pairs[:, 0]
        j = pairs[:, 1]
        C = len(pairs)

        G, g, sx, sy = self._incremental_gram_matrix(X, y)

        A = torch.zeros((C, 3, 3), dtype=torch.float, device=self.device)
        A[:, 0, 0] = n
        A[:, 0, 1] = sx[i]
        A[:, 0, 2] = sx[j]

        A[:, 1, 0] = sx[i]
        A[:, 1, 1] = G[i, i]
        A[:, 1, 2] = G[i, j]

        A[:, 2, 0] = sx[j]
        A[:, 2, 1] = G[j, i]
        A[:, 2, 2] = G[j, j]

        b = torch.zeros((C, 3), dtype=torch.float, device=self.device)
        b[:, 0] = sy
        b[:, 1] = g[i]
        b[:, 2] = g[j]

        try:
            betas, *_ = torch.linalg.lstsq(A, b)
        except:
            print('WARNING: Singular matrix, adding small perturbation')
            I = torch.eye(3, device=A.device).unsqueeze(0).expand(A.shape[0], 3, 3)
            eps = 1e-6 * A.abs().mean(dim=(1,2), keepdim=True)
            betas, *_ = torch.linalg.lstsq(A + eps * I, b)

        self.coef_ = betas[:, [1, 2]].float()
        self.intercept_ = betas[:, 0].float()
        self.i = i
        self.j = j
        self.C = C

        return self

    @ensure_x
    def predict(self, X):
        preds = torch.empty((X.shape[0], self.C), dtype=X.dtype)
        element_size = X.element_size()
        
        start = 0
        while start < X.shape[0]:
            Xb = X[
                start:
                start + (batch_size := self._calculate_batch_size(element_size=element_size, dim=self.C))
            ].to(self.device)

            out = (
                self.intercept_[None, :] +
                Xb[:, self.i] * self.coef_[:, 0] + 
                Xb[:, self.j] * self.coef_[:, 1]
            )

            preds[start:start + batch_size] = out.cpu()
            start += batch_size
            
            torch.cuda.empty_cache()

        return preds

    @ensure_y
    @ensure_x
    def evaluate(self, X, y, metrics=None, return_preds=False, pow_cross=False):
        if metrics is None:
            metrics = self.metrics
        res = {}
        y_pred = self.predict(X)

        if pow_cross:
            y_pred = torch.exp(y_pred)
        
        for metric in metrics:
            if metric == "rmse":
                scores = torch.sqrt(((y_pred - y[:, None])**2).mean(dim=0))
            elif metric == "mse":
                scores = ((y_pred - y[:, None])**2).mean(dim=0)
            elif metric == "mae":
                scores = (y_pred - y[:, None]).abs().mean(dim=0)
            elif metric == "r2":
                y_mean = y.mean(dim=0)
                ss_tot = ((y - y_mean[None, :])**2).sum(dim=0)
                ss_res = ((y_pred - y[:, None])**2).sum(dim=0)
                scores = 1 - ss_res / ss_tot
            elif metric == "mape":
                scores = ((y_pred - y[:, None]).abs() / y[:, None]).mean(dim=0)
            elif metric == "accuracy":
                scores = (y_pred.round() == y[:, None]).mean(dim=0)
            elif metric == "precision":
                scores = (y_pred.round() == y[:, None]).sum(dim=0) / (y_pred.round() == y[:, None]).sum(dim=0)
            elif metric == "recall":
                scores = (y_pred.round() == y[:, None]).sum(dim=0) / (y[:, None] == y[:, None]).sum(dim=0)
            elif metric == "f1":
                precision = (y_pred.round() == y[:, None]).sum(dim=0) / (y_pred.round() == y[:, None]).sum(dim=0)
                recall = (y_pred.round() == y[:, None]).sum(dim=0) / (y[:, None] == y[:, None]).sum(dim=0)
                scores = 2 * (precision * recall) / (precision + recall)
            else:
                raise ValueError(f"Invalid metric: {metric}")

            res[metric] = scores
            
        if return_preds:
            return res, y_pred

        return res


