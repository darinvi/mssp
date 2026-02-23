import networkx as nx
import matplotlib.pyplot as plt


def _to_scalar(x):
    """Convert tensor or int to scalar for use in node IDs."""
    return x.item() if hasattr(x, 'item') else x


def model_graph_to_nx(model_graph, colnames=None):
    """Convert ModelGraph to NetworkX DiGraph."""
    G = nx.DiGraph()

    def add_node(node, parent_id=None):
        if hasattr(node, 'left_child'):  # Node (internal)
            i, j = _to_scalar(node.i), _to_scalar(node.j)
            node_id = f"n_{node.epoch}_{i}_{j}_{node.cross}"
            label = f"×{node.cross}"
        else:  # PrimitiveNode (leaf)
            node_id = f"p_{node.i}_{node.col_i}_{node.fn}"
            col_label = colnames[node.col_i] if colnames and node.col_i < len(colnames) else f"col{node.col_i}"
            label = f"{col_label}\n{node.fn}"

        G.add_node(node_id, label=label)
        if parent_id is not None:
            G.add_edge(parent_id, node_id)

        if hasattr(node, 'left_child'):
            add_node(node.left_child, node_id)
            add_node(node.right_child, node_id)

    add_node(model_graph.head)
    return G


def _tree_layout(G, root, depth=0, x=0.0, dx=1.0):
    """
    Recursively compute tree layout positions.
    Returns dict of node_id -> (x, y).
    """
    pos = {}
    successors = list(G.successors(root))

    if not successors:
        pos[root] = (x, -depth)
        return pos, x + dx

    # Left and right children
    left, right = successors[0], successors[1]
    pos_left, x_mid = _tree_layout(G, left, depth + 1, x, dx / 2)
    pos_right, x_end = _tree_layout(G, right, depth + 1, x_mid, dx / 2)

    pos.update(pos_left)
    pos.update(pos_right)
    pos[root] = ((pos[left][0] + pos[right][0]) / 2, -depth)
    return pos, x_end


def plot_model_graph(model_graph, ax=None, top_k=1, colnames=None):
    """
    Plot a ModelGraph as a tree using NetworkX.

    Parameters
    ----------
    model_graph : ModelGraph or MSSP
        The graph to plot. If MSSP instance, uses model[top_k-1].
    ax : matplotlib Axes, optional
        Axes to draw on. If None, uses current axes.
    top_k : int, optional
        When model_graph is MSSP, which model to plot (1-indexed).
    colnames : list of str, optional
        Column names for primitive nodes. If provided, uses these instead of
        "col0", "col1", etc. (e.g. from DataFrame.columns).
    """
    if hasattr(model_graph, 'model'):
        # MSSP instance - need to build model and get ModelGraph
        model_graph._build_model(top_k)
        model_graph = model_graph.model[top_k - 1]

    G = model_graph_to_nx(model_graph, colnames=colnames)
    root = [n for n in G.nodes() if G.in_degree(n) == 0][0]
    pos, _ = _tree_layout(G, root)

    ax = ax or plt.gca()
    labels = nx.get_node_attributes(G, 'label')

    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        labels=labels,
        node_size=1200,
        node_color='lightblue',
        font_size=8,
        font_weight='bold',
        arrows=True,
        arrowsize=12,
        edge_color='gray',
        connectionstyle='arc3,rad=0',
    )
    ax.axis('off')
    return ax


class MSSPPlot:
    def plot_graph(self, top_k=1, ax=None, colnames=None):
        """
        Plot the MSSP model graph as a tree.

        Parameters
        ----------
        top_k : int
            Which model to plot (1-indexed).
        ax : matplotlib Axes, optional
            Axes to draw on. If None, creates a new figure.
        colnames : list of str, optional
            Column names for primitive nodes (e.g. df.columns.tolist()).
            If provided, uses these instead of "col0", "col1", etc.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        return plot_model_graph(self, ax=ax, top_k=top_k, colnames=colnames)
