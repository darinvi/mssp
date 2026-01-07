class Node:
    def __init__(self, epoch, history, i):
        self.is_primitive = epoch == 0
        
        if self.is_primitive:
            self._build_primitive_node(history, i)
            return

        self.left_child = Node(epoch - 1, history, i)
        self.right_child = Node(epoch - 1, history, i)
        self._build_node(history, i)


    def _build_node(self, history, i):
        pass

    def _build_primitive_node(self, history, i):
        pass

    @staticmethod
    def build_node(history, i):
        max_ep = max([h['epoch'] for h in history])
        return Node(max_ep, history, i)
    
    