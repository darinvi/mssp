from src.ols import LinearRegression
import torch



class DataManager:
    def __init__(self, return_params=False):
        self.normalizer = Normalizer()
        self.primitives = Primitives()

    def fit(self, X, y):
        X = self.normalizer.fit_transform(X)
        X, params = self.primitives.fit_transform(X, y)
        if X.isnan().any():
            raise Exception("NaNs in X")
        return X, params

    def transform(self, X, apply_primitives=True, clip=False):
        X = self.normalizer.transform(X, clip)
        if apply_primitives:
            X = self.primitives.transform(X)
        if X.isnan().any():
            raise Exception("NaNs in X")
        return X

class Normalizer:
    def __init__(self, eps=1e-6, out_range=None):
        self.eps = eps
        self.out_range = out_range or (eps, 1 - eps)

    def fit(self, X):
        if hasattr(self, 'fitted'):
            raise Exception("Already fitted")
        self.min = X.min(dim=0)[0]
        self.max = X.max(dim=0)[0]
        self.fitted = True
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X, clip=False):
        if not hasattr(self, 'fitted') or not self.fitted:
            raise Exception("Normalizer should be fitted before applying transformations.")
        
        '''
        If incoming data is outside of the training range, outcomes might be unpredictable combined with logarithms.
        '''
        if clip:
            X = torch.clamp(X, min=self.min, max=self.max)

        low, high = self.out_range
        return (X - self.min) / (self.max - self.min + self.eps) * (high - low) + low

    def inverse_transform(self, X):
        low, high = self.out_range
        return (X - low) / (high - low) * (self.max - self.min + self.eps) + self.min


class Primitives:
    def __init__(self, apply_correction: bool = True):
        _lin = lambda x: x
        _log = lambda x: torch.log(x)
        _exp = lambda x: torch.exp(x)
        _rec = lambda x: 1/x
        self.functions = [
            (self.lin, _lin, _lin),
            (self.lgn, _log, _lin),
            (self.xpy, _lin, _exp),
            (self.pow, _log, _exp),
            (self.rex, _rec, _lin),
            (self.rey, _lin, _rec),
            (self.sqr, _lin, lambda x: x**2),
            (self.snx, lambda x: torch.sin(x), _lin),
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

    def transform(self, X):
        res = torch.zeros(X.shape[0], X.shape[1] * len(self.functions), dtype=torch.float)
        for i, models in enumerate(self.models):
            for j, (m, fn_in, fn_out) in enumerate(models):
                y = m.predict(fn_in(X[:, i]))
                res[:, i * len(self.functions) + j] = fn_out(y)
        return res

    def fit_transform(self, X, y):
        X, y = Primitives._validate_input(X, y)

        res = torch.zeros(X.shape[0], X.shape[1] * len(self.functions), dtype=torch.float)
        params = []
        models = []
        for i in range(X.shape[1]):
            feature_models = []
            for j, (func, fn_in, fn_out) in enumerate(self.functions):
                tensor, coef, intercept, model = func(X[:, i], y)
                res[:, i * len(self.functions) + j] = tensor
                params.append((coef, intercept, func.__name__, i))
                feature_models.append((model, fn_in, fn_out))
            models.append(feature_models)

        self.models = models

        return res, params
    

    '''
    Y = a0 + a1*X
    '''
    def lin(self, x, y):
        model = LinearRegression()
        p = model.fit_predict(x, y)
        return p, model.coef_, model.intercept_, model


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
        return p, model.coef_, model.intercept_, model


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
        return torch.exp(p), model.coef_, model.intercept_, model

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
        return torch.exp(p), model.coef_, model.intercept_, model


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
        return p, model.coef_, model.intercept_, model


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
        return 1/p, model.coef_, model.intercept_, model


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
        return p ** 2, model.coef_, model.intercept_, model


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
        return p, model.coef_, model.intercept_, model

