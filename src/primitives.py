from src import regression as lr
import torch
from src.manager import Manager


class apply_shift:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        def wrapper(x, y):
            x_corrected, corrections_x = instance.shift_x(x) if instance.apply_correction else (x, [])
            y_corrected, corrections_y = instance.shift_y(y) if instance.apply_correction else (y, [])
            return self.func(instance, x_corrected, y_corrected, corrections_x + corrections_y)
        return wrapper


class Primitives:
    def __init__(self, manager: Manager, apply_correction: bool = True):
        self.apply_correction = apply_correction
        self.manager = manager
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


    def __call__(self, x, y):
        res = torch.zeros(len(x), len(self.functions))
        for i, func in enumerate(self.functions):
            tensor, coef, intercept, corrections = func(x, y)
            res[:, i] = tensor
        return res
    

    def get_primitives(self, x, y):
        return self(x, y)


    def shift_x(self, x):
        return x, []


    def shift_y(self, y):
        return y, []


    '''
    Y = a0 + a1*X
    '''
    @apply_shift
    def lin(self, x, y, corrections):
        model = lr.LinearRegression()
        p = model.fit_predict(x, y)
        return p, model.coef_, model.intercept_, corrections


    '''
    Y = a0 + a1*ln(X)
    Z = ln(X)
    Y = a0 + a1*Z
    '''
    @apply_shift
    def lgn(self, x, y, corrections):
        model = lr.LinearRegression()
        p = model.fit_predict(
            torch.log(x),
            y
        )
        return p, model.coef_, model.intercept_, corrections


    '''
    Y = e ** (a0 + a1*X)
    Q = ln(Y)
    Q = a0 + a1*X
    '''
    @apply_shift
    def xpy(self, x, y, corrections):
        model = lr.LinearRegression()
        p = model.fit_predict(
            x, 
            torch.log(y)
        )
        return torch.exp(p), model.coef_, model.intercept_, corrections

    '''
    Y = a0 * X^a1
    Z = ln(X)
    ln(Y) = ln(a0) + a1*Z
    '''
    @apply_shift
    def pow(self, x, y, corrections):
        model = lr.LinearRegression()
        p = model.fit_predict(
            torch.log(x), 
            torch.log(y)
        )
        return torch.exp(p), model.coef_, model.intercept_, corrections


    '''
    Y = a0 + a1/X
    Z = 1/X
    Y = a0 + a1*Z
    '''
    @apply_shift
    def rex(self, x, y, corrections):
        model = lr.LinearRegression()
        p = model.fit_predict(
            1/x,
            y
        )
        return p, model.coef_, model.intercept_, corrections


    '''
    Y = 1 / (a0 + a1*X)
    Q = 1/Y
    Q = a0 + a1*X
    '''
    @apply_shift
    def rey(self, x, y, corrections):
        model = lr.LinearRegression()
        p = model.fit_predict(
            x, 
            1/y
        )
        return 1/p, model.coef_, model.intercept_, corrections


    '''
    Y = (a0 + a1*X)^2
    Q = sqrt(Y)
    Q = a0 + a1*X
    '''
    @apply_shift
    def sqr(self, x, y, corrections):
        model = lr.LinearRegression()
        p = model.fit_predict(
            x, 
            torch.sqrt(y)
        )
        return p ** 2, model.coef_, model.intercept_, corrections


    '''
    Y = a0 + a1*sin(X)
    Z = sin(X)
    Y = a0 + a1*Z
    '''
    @apply_shift
    def snx(self, x, y, corrections):
        model = lr.LinearRegression()
        p = model.fit_predict(
            torch.sin(x),
            y
        )
        return p, model.coef_, model.intercept_, corrections
