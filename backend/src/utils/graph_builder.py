import numpy as np
from skimage.morphology import skeletonize
import networkx as nx

def get_skeleton_from_mask(mask):
    """Convert binary mask (0/255 or boolean) to skeleton (bool ndarray)."""
    if mask.dtype != bool:
        binary = mask > 127
    else:
        binary = mask
    return skeletonize(binary).astype(np.uint8)


def _neighbors(y, x, shape):
    H, W = shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            next_y, next_x = y + dy, x + dx
            if 0 <= next_y < H and 0 <= next_x < W:
                yield next_y, next_x


def build_graph_from_skeleton(skel):
    """Build a NetworkX graph from a 1-pixel-wide skeleton image.

    Nodes are placed at endpoints and junctions (degree != 2). Edges
    follow skeleton paths between nodes and carry a `pixels` list and `length`.
    Returns an undirected NetworkX Graph with nodes keyed by (y,x) tuples.
    """
    if skel.dtype != np.uint8:
        skel = skel.astype(np.uint8)

    H, W = skel.shape
    pts = set(zip(*np.where(skel > 0)))

    # degree for skeleton pixels
    degree = {}
    for p in pts:
        y, x = p
        cnt = 0
        for next_y, next_x in _neighbors(y, x, (H, W)):
            if (next_y, next_x) in pts:
                cnt += 1
        degree[p] = cnt

    # nodes are endpoints (deg==1) or junctions (deg>2)
    nodes = {p for p, d in degree.items() if d == 1 or d > 2}

    # If closed loops exist (all deg==2), pick an arbitrary pixel as a node
    if len(nodes) == 0 and len(pts) > 0:
        nodes.add(next(iter(pts)))

    G = nx.Graph()

    # add nodes to graph
    for n in nodes:
        G.add_node(n)

    visited = set()

    def walk_path(start, neighbor):
        """Walk from start pixel into neighbor until another node is reached."""
        path = [start]
        cur = neighbor
        prev = start
        path_visited = {start, neighbor}  # FIX: track full path to prevent infinite loops on cyclic skeletons
        while True:
            path.append(cur)
            visited.add(cur)
            if cur in nodes and cur != start:
                return cur, path
            # find next neighbors excluding previous and already-visited path pixels
            nexts = [p for p in _neighbors(cur[0], cur[1], (H, W)) if p in pts and p != prev and p not in path_visited]
            if not nexts:
                # dead end or cycle closed
                return cur, path
            # advance
            next_px = nexts[0]
            path_visited.add(next_px)
            prev, cur = cur, next_px

    # For each node, start walking down each outgoing branch
    for n in nodes:
        y, x = n
        for nb in _neighbors(y, x, (H, W)):
            if nb in pts and nb not in visited:
                target, path = walk_path(n, nb)
                if target is None:
                    continue
                # determine edge endpoints (n and target)
                a, b = tuple(n), tuple(target)
                if a == b:
                    continue
                if G.has_edge(a, b):
                    # if edge exists, maybe keep the shorter representation
                    continue
                G.add_edge(a, b, pixels=path, length=len(path))

    return G


def simplify_graph(G, min_length=2):
    """Remove tiny spurious edges (optionally) by pruning edges shorter than min_length."""
    H = G.copy()
    for u, v, d in list(G.edges(data=True)):
        if d.get("length", 0) < min_length:
            H.remove_edge(u, v)
    # remove isolated nodes
    for n in list(H.nodes()):
        if H.degree(n) == 0:
            H.remove_node(n)
    return H

