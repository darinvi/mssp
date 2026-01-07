from src.ols import LinearRegression
import torch


class Normalizer:
    def __init__(self, eps=1e-6, out_range=None):
        self.eps = eps
        self.out_range = out_range or (eps, 1 - eps)

    def fit(self, X):
        self.min = X.min(dim=0)[0]
        self.max = X.max(dim=0)[0]
        return self.transform(X)

    def transform(self, X):
        low, high = self.out_range
        return (X - self.min) / (self.max - self.min + self.eps) * (high - low) + low

    def inverse_transform(self, X):
        low, high = self.out_range
        return (X - low) / (high - low) * (self.max - self.min + self.eps) + self.min


class Primitives:
    def __init__(self, apply_correction: bool = True):
        self.functions = [
            self.lin,
            self.lgn,
            self.xpy,
            self.pow,
            self.rex,
            self.rey,
            self.sqr,
            self.snx,
        ]


    @staticmethod
    def _validate_input(X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float)
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y, dtype=torch.float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X must be a 1D or 2D tensor, received {X.ndim}D")
        return X, y


    def transform(self, X, y, return_params: bool = False):
        X, y = Primitives._validate_input(X, y)

        res = torch.zeros(X.shape[0], X.shape[1] * len(self.functions), dtype=torch.float)
        params = []
        for i in range(X.shape[1]):
            for j, func in enumerate(self.functions):
                tensor, coef, intercept = func(X[:, i], y)
                res[:, i * len(self.functions) + j] = tensor
                if return_params:
                    params.append((coef, intercept))

        if return_params:
            return res, params

        return res
    

    '''
    Y = a0 + a1*X
    '''
    def lin(self, x, y):
        model = LinearRegression()
        p = model.fit_predict(x, y)
        return p, model.coef_, model.intercept_


    '''
    Y = a0 + a1*ln(X)
    Z = ln(X)
    Y = a0 + a1*Z
    '''
    def lgn(self, x, y):
        model = LinearRegression()
        p = model.fit_predict(
            torch.log(x),
            y
        )
        return p, model.coef_, model.intercept_


    '''
    Y = e ** (a0 + a1*X)
    Q = ln(Y)
    Q = a0 + a1*X
    '''
    def xpy(self, x, y):
        model = LinearRegression()
        p = model.fit_predict(
            x, 
            torch.log(y)
        )
        return torch.exp(p), model.coef_, model.intercept_

    '''
    Y = a0 * X^a1
    Z = ln(X)
    ln(Y) = ln(a0) + a1*Z
    '''
    def pow(self, x, y):
        model = LinearRegression()
        p = model.fit_predict(
            torch.log(x), 
            torch.log(y)
        )
        return torch.exp(p), model.coef_, model.intercept_


    '''
    Y = a0 + a1/X
    Z = 1/X
    Y = a0 + a1*Z
    '''
    def rex(self, x, y):
        model = LinearRegression()
        p = model.fit_predict(
            1/x,
            y
        )
        return p, model.coef_, model.intercept_


    '''
    Y = 1 / (a0 + a1*X)
    Q = 1/Y
    Q = a0 + a1*X
    '''
    def rey(self, x, y):
        model = LinearRegression()
        p = model.fit_predict(
            x, 
            1/y
        )
        return 1/p, model.coef_, model.intercept_


    '''
    Y = (a0 + a1*X)^2
    Q = sqrt(Y)
    Q = a0 + a1*X
    '''
    def sqr(self, x, y):
        model = LinearRegression()
        p = model.fit_predict(
            x, 
            torch.sqrt(y)
        )
        return p ** 2, model.coef_, model.intercept_


    '''
    Y = a0 + a1*sin(X)
    Z = sin(X)
    Y = a0 + a1*Z
    '''
    def snx(self, x, y):
        model = LinearRegression()
        p = model.fit_predict(
            torch.sin(x),
            y
        )
        return p, model.coef_, model.intercept_

class DataManager:
    def __init__(self, return_params=False):
        self.normalizer = Normalizer()
        self.primitives = Primitives()

    def fit(self, X, y):
        X = self.normalizer.fit(X)
        X = self.primitives.transform(X, y)
        if X.isnan().any():
            raise Exception("NaNs in X")
        return X

    def transform(self, X, y):
        X = self.normalizer.transform(X)
        X = self.primitives.transform(X, y)
        if X.isnan().any():
            raise Exception("NaNs in X")
        return X