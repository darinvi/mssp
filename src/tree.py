import torch

class Node:
    def __init__(self, epoch, history, i):
        self.epoch = epoch
        self._build_node(history, i)

    def _build_node(self, history, i):
            # TODO handle primitives
        lin = [h for h in history if h['epoch'] == self.epoch and h['type'] == 'lin_cross_selection'][0]
        pow = [h for h in history if h['epoch'] == self.epoch and h['type'] == 'pow_cross_selection'][0]
        best_i = [h for h in history if h['epoch'] == self.epoch and h['type'] == 'best_selection'][0]['mask'][i]

        self.coef_ = torch.cat([lin['coef'], pow['coef']], dim=0)[best_i]
        self.intercept_ = torch.cat([lin['intercept'], pow['intercept']], dim=0)[best_i]

        self.i = torch.cat([lin['i'], pow['i']])[best_i]
        self.j = torch.cat([lin['j'], pow['j']])[best_i]

        self.left_child = Node(self.epoch - 1, history, self.i, dim=0) if self.epoch > 0 else PrimitiveNode(self.i)
        self.right_child = Node(self.epoch - 1, history, self.j, dim=0) if self.epoch > 0 else PrimitiveNode(self.j)

    def predict(self, X):
        pass

    @staticmethod
    def build_node(history, i):
        max_ep = max([h['epoch'] for h in history])
        return Node(max_ep, history, i)
    

class PrimitiveNode:
    def __init__(self, i):
        self.i = i

    def predict(self, X):
        pass