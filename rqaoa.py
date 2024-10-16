import numpy as np
import networkx as nx
import random
import itertools
import copy
import time
import os
import subprocess
from typing import Tuple
from typing import List
from typing import Dict


def get_rqaoa_result(
    edges: Tuple[Tuple[int, int]],
    edge_weights: Tuple[float],
    brute_force_threshold: int = 10,
    edge_noise: float = 1e-5,
    nb_div: int = 5,
    div_level: int = 3,
    fninp: str = 'graphdata.txt',
    fnhyp: str = 'hyps.txt',
    fnout: str = 'result.txt',
) -> Tuple[str, float, float]:
    
    dir = os.getcwd() + '/'
    graph_data_file = open(fninp, 'wt')
    nb_nodes = int(np.max(list(set(np.ravel(edges))))) + 1
    graph_data_file.write(str(nb_nodes)+' '+str(len(edges))+'\n')
    for (p, q), w in zip(edges, edge_weights):
        graph_data_file.write(str(p+1)+' '+str(q+1)+' '+str(w)+'\n')
    graph_data_file.close()

    with open(fnhyp, 'w') as f:
        print(str(brute_force_threshold) + '     ................ brute_force_threshold : threshold of No. nodes to start brute-force search', file=f)
        print(str(edge_noise) + '  ................ edge_noise            : edge noise', file=f)
        print(str(nb_div) + '      ................ nb_div                : No. mesh for grid search', file=f)
        print(str(div_level) + '      ................ div_level             : mesh refinement level', file=f)

    while True:
        cp = subprocess.run(dir+'rqaoa '+fninp+' '+fnhyp+' '+fnout, shell=True)
        if cp.returncode == 0:
            break
            
    with open(fnout) as f:
        dat = [l for l in f]
        bits = dat[0][:-1]
        cut = float(dat[1])
        time = float(dat[2])
        
    return bits, cut, time


def get_reduced_candits(
    edge_energy_dict: Dict[Tuple[int, int], float],
    edge_energy_thresh: float,
) -> Tuple[List[Tuple[int, int]], Tuple[int]]:
    
    if edge_energy_thresh is None:
        w_max = float('-inf')
        for (p, q), w in edge_energy_dict.items():
            if np.abs(w) > w_max:
                weighted_edges = [(p, q, np.abs(w))]
                w_max = np.abs(w)
    else:
        weighted_edges = [(p, q, abs(w)) for (p, q), w in edge_energy_dict.items() if abs(w) >= edge_energy_thresh]
        if len(weighted_edges) == 0:
            emax = 0.
            for (p, q), w in edge_energy_dict.items():
                if np.abs(w) > emax:
                    weighted_edges = [(p, q, abs(w))]
                    emax = np.abs(w)
        
    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edges)
    S = [G.subgraph(c) for c in nx.connected_components(G)]
    
    candits = []
    parities = []
    
    for subgraph in S:
        mst = nx.maximum_spanning_tree(subgraph)
        root = random.choice(list(set(np.ravel(mst.edges()))))
        bfs_tree = nx.bfs_tree(mst, root)
        for e in list(bfs_tree.edges())[::-1]:
            candits += [(e[1], e[0])]
            if (e[0], e[1]) in edge_energy_dict.keys():
                energy = edge_energy_dict[(e[0], e[1])]
            else:
                energy = edge_energy_dict[(e[1], e[0])]
            parities += [2 * int(energy > 0.) - 1]

    return [tuple(c) for c in candits], tuple(parities)


def get_new_graph(
    edges: Tuple[Tuple[int, int]],
    edge_weights: Tuple[float],
    candidate: Tuple[int, int],
    sgn_edge_energy: int,
) -> Tuple[Tuple[Tuple[int, int]], Tuple[float], Tuple[int]]:
    
    edge_dict = {}

    for e, w in zip(edges, edge_weights):

        if set(e) == set(candidate):

            pass
        
        elif e == (candidate[0], candidate[0]):
            
            edge = (candidate[1], candidate[1])
            weight = sgn_edge_energy * w
            
            if edge in edge_dict.keys():

                edge_dict.update({edge: edge_dict[edge] + weight})

            else:

                edge_dict.update({edge: weight})

        else:

            if (candidate[0] in e) and (candidate[1] not in e):

                p2 = candidate[1]
                q2 = list(set(e) - set([candidate[0]]))[0]
                edge = (min(p2, q2), max(p2, q2))
                weight = sgn_edge_energy * w

            else:

                edge = (min(e), max(e))
                weight = w

            if edge in edge_dict.keys():

                edge_dict.update({edge: edge_dict[edge] + weight})

            else:

                edge_dict.update({edge: weight})

    edges_reduced = []
    edge_weights_reduced = []

    for k, v in edge_dict.items():

        edges_reduced += [k]
        edge_weights_reduced += [v]

    nb_nodes = len(set(np.ravel(edges_reduced)))
    node_idx_map = list(range(nb_nodes))

    if len(edges_reduced) > 0:

        if len(set(np.ravel(edges_reduced))) != np.max(np.ravel(edges_reduced)) + 1:

            nodes = list(sorted(set(np.ravel(edges_reduced))))
            replace_target = sorted(set(nodes) - set(range(len(nodes))))
            replace_candit = sorted(set(range(len(nodes))) - set(nodes))

            for target, candit in zip(replace_target, replace_candit):

                node_idx_map[candit] = target

            edges_replaced = []

            for p, q in edges_reduced:

                if p in replace_target:

                    p_replaced = replace_candit[replace_target.index(p)]

                else:

                    p_replaced = p

                if q in replace_target:

                    q_replaced = replace_candit[replace_target.index(q)]

                else:

                    q_replaced = q

                edges_replaced += [(p_replaced, q_replaced)]

            edges_reduced = edges_replaced
            
    return tuple(edges_reduced), tuple(edge_weights_reduced), tuple(node_idx_map)


def get_best_cut(
    nb_nodes: int,
    edges: Tuple[Tuple[int, int]],
    edge_weights: Tuple[float],
    map_list: List[List[int]],
    candidate_list: List[Tuple[int]],
    sgn_edge_energy_list: List[float],
) -> Tuple[float, List[str]]:

    bits_list = get_bits_list(nb_nodes, map_list, candidate_list, sgn_edge_energy_list)
    bits_list_filled = []
    for bits in bits_list:
        nb_unknown = bits.count('*')
        for v in itertools.product(['0', '1'], repeat=nb_unknown):
            b = copy.deepcopy(bits)
            for vv in v:
                b = b[:b.find('*')] + vv + b[b.find('*')+1:]
            bits_list_filled += [b]
            
    bits = np.array([[int(bb) for bb in b[::-1]] for b in bits_list_filled])
    edges_1 = []
    edges_2 = []
    edge_weights_1 = []
    edge_weights_2 = []
    for e, w in zip(edges, edge_weights):
        if e[0] != e[1]:
            edges_1 += [e]
            edge_weights_1 += [w]
        else:
            edges_2 += [e]
            edge_weights_2 += [w]

    cut_list = np.sum([w * (bits[:, p] != bits[:, q]) for (p, q), w in zip(edges_1, edge_weights_1)], 0) + np.sum([w * (bits[:, p] == 1) for (p, q), w in zip(edges_2, edge_weights_2)], 0)
    cut_max = cut_list.max()
    gs_states = np.array(bits_list_filled)[cut_list == cut_max]
    
    return cut_max, gs_states


def get_bits_list(
    nb_nodes: int,
    map_list: List[List[int]],
    candidate_list: List[Tuple[int]],
    sgn_edge_energy_list: List[float],
) -> List[str]:

    map_list_expanded = [list(range(nb_nodes))] + list(map_list)
    cut_edges = []

    for i in range(len(map_list)):
        
        p, q = candidate_list[i]

        node_idx = {}
        for j in range(nb_nodes):
            node_idx.update({j:j})

        for ml in map_list_expanded[:i+1]:
            for j, idx in enumerate(ml):
                if j != idx:
                    node_idx[j] = idx
        
        while True:
            break_flag = True
            for k, v in node_idx.items():
                if node_idx[k] != node_idx[v]:
                    node_idx.update({k: node_idx[v]})
                    break_flag = False
            if break_flag:
                break
                
        cut_edges += [(node_idx[p], node_idx[q], sgn_edge_energy_list[i])]
    
    edges = []
    for p, q, _ in cut_edges:
        if (p, q) in edges:
            print((p,q),edges)
            raise
        if (q, p) in edges:
            print((p,q),edges)
            raise
        edges += [(p, q)]
        
    G = nx.Graph()
    G.add_weighted_edges_from(cut_edges)
    S = [G.subgraph(c) for c in nx.connected_components(G)]
    
    bits_list = ['*' * nb_nodes]
    
    for s in S:
        mst = nx.maximum_spanning_tree(s)
        root = np.min(np.ravel(mst.edges()))
        bfs_tree = nx.bfs_tree(mst, root)
        
        edges = list(bfs_tree.edges)
        
        parities = []
        for p, q in edges:
            for u, v, d in mst.edges(data=True):
                if set([u, v]) == set([p, q]):
                    parities += [int(d['weight'])]
            
        for i, ((p, q), parity) in enumerate(zip(edges, parities)):
            if i == 0:
                bits_list_a = bits_list.copy()
                bits_list_b = bits_list.copy()
                
                bits_list_a2 = []
                for bits in bits_list_a:
                    if parity == 1:
                        bits = bits[:p] + '0' + bits[p+1:]
                        bits = bits[:q] + '0' + bits[q+1:]
                    elif parity == -1:
                        bits = bits[:p] + '0' + bits[p+1:]
                        bits = bits[:q] + '1' + bits[q+1:]
                    else:
                        raise
                    bits_list_a2 += [bits]
                    
                bits_list_b2 = []
                for bits in bits_list_b:
                    if parity == 1:
                        bits = bits[:p] + '1' + bits[p+1:]
                        bits = bits[:q] + '1' + bits[q+1:]
                    elif parity == -1:
                        bits = bits[:p] + '1' + bits[p+1:]
                        bits = bits[:q] + '0' + bits[q+1:]
                    bits_list_b2 += [bits]
                    
                bits_list = (bits_list_a2 + bits_list_b2).copy()
            else:
                foo = []
                for bits in bits_list:
                    if bits[p] == '*':
                        if parity == 1:
                            b = bits[q]
                        if parity == -1:
                            b = str(abs(int(bits[q])-1))
                        else:
                            raise
                        bits = bits[:p] + b + bits[p+1:]
                    elif bits[q] == '*':
                        if parity == 1:
                            b = bits[p]
                        elif parity == -1:
                            b = str(abs(int(bits[p])-1))
                        else:
                            raise
                        bits = bits[:q] + b + bits[q+1:]
                    else:
                        raise
                    foo += [bits]
                bits_list = foo.copy()
                    
    bits_list = [bits[::-1] for bits in bits_list]
    
    return bits_list