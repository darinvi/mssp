import torch


class ModelGraph:
    def __init__(self, history, i):
        max_ep = max([h['epoch'] for h in history])
        self.nodes = {}
        self.head = Node(max_ep, history, i, self)
        self.nodes["head"] = self.head

    def predict(self, X):
        return torch.vmap(self.head.predict)(X)


class Node:
    def __init__(self, epoch, history, i, graph):
        self.epoch = epoch
        self.graph = graph
        self._build_node(history, i)
        self.key = (epoch, i, self.cross)

    def _build_node(self, history, i):
        lin = [h for h in history if h['epoch'] == self.epoch and h['type'] == 'lin_cross_selection'][0]
        pow = [h for h in history if h['epoch'] == self.epoch and h['type'] == 'pow_cross_selection']
        if pow:
            pow = pow[0]
        best_i = [h for h in history if h['epoch'] == self.epoch and h['type'] == 'best_selection'][0]['mask'][i]

        self.coef_ = torch.cat([lin['coef'], pow['coef']], dim=0)[best_i] if pow else lin['coef'][best_i]
        self.intercept_ = torch.cat([lin['intercept'], pow['intercept']], dim=0)[best_i] if pow else lin['intercept'][best_i]

        self.i = torch.cat([lin['i'], pow['i']], dim=0)[best_i] if pow else lin['i'][best_i].item()
        self.j = torch.cat([lin['j'], pow['j']], dim=0)[best_i] if pow else lin['j'][best_i].item()

        self.cross = 'lin' if best_i < len(lin['mask']) else 'pow'
        self.fn_in = (lambda x: x) if self.cross == 'lin' else (lambda x: torch.log(x))
        self.fn_out = (lambda x: x) if self.cross == 'lin' else (lambda x: torch.exp(x))

        if self.epoch > 0:
            lkey = (self.epoch - 1, self.i, self.cross)
            rkey = (self.epoch - 1, self.j, self.cross)
            lchild = self.graph.nodes.get(
                lkey,
                Node(self.epoch - 1, history, self.i, self.graph)
            )
            rchild = self.graph.nodes.get(
                rkey,
                Node(self.epoch - 1, history, self.j, self.graph)
            )
        else:
            lkey = (self.i)
            rkey = (self.j)
            lchild = self.graph.nodes.get(
                lkey,
                PrimitiveNode(history, self.i)
            )
            rchild = self.graph.nodes.get(
                rkey,
                PrimitiveNode(history, self.j)
            )

        self.graph.nodes[lkey] = lchild
        self.graph.nodes[rkey] = rchild
        self.left_child = lchild
        self.right_child = rchild

    def predict(self, x):
        f1 = self.fn_in(self.left_child.predict(x))
        f2 = self.fn_in(self.right_child.predict(x))
        y = self.coef_[0] * f1 + self.coef_[1] * f2 + self.intercept_
        return self.fn_out(y)


class PrimitiveNode:
    def __init__(self, history, i):
        self.i = i
        self.key = i
        self._build_node(history)

    def _build_node(self, history):
        params = [h for h in history if h['type'] == 'primitives'][0]
        self.coef_, self.intercept_, self.fn, self.col_i = params['params'][self.i]
        _lin = lambda x: x
        _log = lambda x: torch.log(x)
        _exp = lambda x: torch.exp(x)
        _rec = lambda x: 1/x
        self.fn_in, self.fn_out = {
            'lin': (_lin, _lin),
            'lgn': (_log, _lin),
            'xpy': (_lin, _exp),
            'pow': (_log, _exp),
            'rex': (_rec, _lin),
            'rey': (_lin, _rec),
            'sqr': (_lin, lambda x: x**2),
            'snx': (lambda x: torch.sin(x), _lin),
        }.get(self.fn)

    def predict(self, x):
        y = self.coef_ * self.fn_in(x[self.col_i]) + self.intercept_
        return self.fn_out(y)
