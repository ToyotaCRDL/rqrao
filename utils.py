import numpy as np
import os
import random
import networkx as nx
from typing import Tuple
from typing import Optional


def get_2d_toric_graph(
    nb_grids: int,
    extra: bool=False,
) -> Tuple[Tuple[int, int]]:

    edges = []
    for i in range(nb_grids):
        for j in range(nb_grids):
            k = i * nb_grids + j
            if j != nb_grids - 1:
                if not (k, k + 1) in edges:
                    edges += [(k, k + 1)]
            else:
                if not (k - nb_grids + 1, k) in edges:
                    edges += [(k - nb_grids + 1, k)]
            if i != nb_grids - 1:
                if not (k, k + nb_grids) in edges:
                    edges += [(k, k + nb_grids)]
            else:
                if not (j, k) in edges:
                    edges += [(j, k)]
                    
            if extra:
                edges += [(k, nb_grids * nb_grids)]
                    
    return tuple(sorted(edges))


def get_graph(
    weight_type: str,
    *,
    nb_nodes: Optional[int]=None,
    degree: Optional[int]=None,
    density: Optional[float]=None,
    seed: Optional[int]=None,
) -> Tuple[Tuple[int, int], np.ndarray]:
    
    if seed is not None:
        random.seed(seed)
        seed_numpy = random.randint(1, 4294967295)
        seed_os_py = random.randint(1, 4294967295)
        np.random.seed(seed_numpy)
        os.environ['PYTHONHASHSEED'] = str(seed_os_py)
 
    if degree is None:

        itr = 0
        while True:
            G = nx.erdos_renyi_graph(nb_nodes, density)
            nb_edges_cand = np.floor(density * nb_nodes * (nb_nodes - 1) * .5)
            if nx.is_connected(G) and (len(G.edges()) in [nb_edges_cand, nb_edges_cand+1]):
                break
            else:
                itr += 1
                if itr == 10000:
                    raise

    else:

        G = nx.random_regular_graph(degree, nb_nodes, seed=seed)

    edges = tuple([e for e in G.edges()])

    if weight_type == '1':
        edge_weights = np.ones(len(edges))
    elif weight_type == 'pm1':
        edge_weights = np.array(random.choices([-1, 1], k=len(edges)))
    elif weight_type == 'random':
        edge_weights = np.round(2. * np.random.random(len(edges)) - 1., 3)
        
    return edges, edge_weights


def read_graph(
    fname: str,
) -> Tuple[int, Tuple[Tuple[int, int]], np.ndarray]:
    
    edges = []
    edge_weights = []
    with open(os.path.dirname(os.path.abspath(__file__))+'/data/'+fname) as f:
        for line in f:
            p, q, w = line.split(' ')
            if w == '\n':
                nb_nodes = int(p)
                nb_edges = int(q)
            else:
                edges += [(int(p)-1, int(q)-1)]
                edge_weights += [int(w)]
                
    return nb_nodes, edges, edge_weights


def get_cut(
    bits: str,
    edges: Tuple[Tuple[int, int]],
    edge_weights: np.ndarray,
) -> float:

    cut = 0.
    for (p, q), w in zip(edges, edge_weights):
        if p == q:
            if bits[::-1][p] == '1':
                cut += w
        else:
            if bits[::-1][p] != bits[::-1][q]:
                cut += w
            
    return cut