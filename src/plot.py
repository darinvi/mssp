class MSSPPlot:
    def _build_graph(self, top_k=1):
        i = top_k - 1
        head = self.model[i]
        levels = [(
            (head.epoch, head.cross, head.pos), 
            (head.left_child.epoch, head.left_child.cross, head.left_child.pos), 
            (head.right_chile.epoch, head.right_chile.cross, head.right_chile.pos)
        )]
        pass