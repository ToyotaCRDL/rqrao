import numpy as np
from sklearn.utils import shuffle
from typing import Tuple
from typing import Optional
from typing import List


def get_groups(
    edges: Tuple[Tuple[int, int]],
    *,
    max_member: int = 3,
    ldf: bool = True,
):

    edge_dict = {}.fromkeys(set(np.ravel(edges)), [])
    for p, q in edges:
        edge_dict.update({p: edge_dict[p] + [q]})
        edge_dict.update({q: edge_dict[q] + [p]})
        
    nb_nodes = len(set(np.ravel(edges)))
    
    if ldf:
        
        node_degree = np.array([(np.ravel(edges)==i).sum() for i in sorted(set(np.ravel(edges)))])
        vj_list = []
        for d in sorted(list(set(node_degree)))[::-1]:
            vj_list += shuffle(np.array(range(nb_nodes))[node_degree==d].tolist())
            
    else:
        
        vj_list = shuffle(np.array(range(nb_nodes)))
        
    groups = []
        
    for j in range(nb_nodes):

        vj = vj_list[j]

        is_contained = False

        for i, g in enumerate(groups):

            is_candidate = True

            for u in g:
                if vj in edge_dict[u]:
                    is_candidate = False
                    break

            if is_candidate and len(groups[i]) < max_member:
                groups[i] += [vj]
                is_contained = True
                break

        if not is_contained:
            groups += [[vj]]
            
    return groups


def get_qrac_hamiltonian(
    edges: Tuple[Tuple[int, int]],
    edge_weights: Tuple[float],
    *,
    mode: str = '31',
    max_member: int = 3,
    paulis: Optional[str] = None,
    ldf: bool = True,
    nb_groups: Optional[int] = None,
    colored_vertices: Optional[List[str]] = None,
):
    
    nb_nodes = len(set(np.ravel(edges)))
    
    if paulis is None:

        paulis = ['X', 'Y', 'Z']

    else:

        flag = False
        flag += (mode == '11') and (len(paulis) > 1)
        flag += (mode == '21') and (len(paulis) > 2)
        flag += (mode == '31') and (len(paulis) > 3)
        if flag:
            raise

        paulis = [p for p in paulis]

    if colored_vertices is None:

        for itr in range(100):

            groups = get_groups(edges, max_member=min(len(paulis), max_member), ldf=ldf)

            if nb_groups is None:

                break

            else:

                if len(groups) <= nb_groups:

                    while len(groups) < nb_groups:

                        idx = random.choice(range(len(groups)))
                        if len(groups[idx]) > 1:
                            nb_samples = random.choice(range(1, len(groups[idx])))
                            news = random.sample(groups[idx], k=nb_samples)
                            foo = []
                            bar = []
                            for m in groups[idx]:
                                if m in news:
                                    bar += [m]
                                else:
                                    foo += [m]
                            groups[idx] = foo
                            groups += [bar]                    

                    break

        else:

            raise('Number of qubits (n=' + str(len(groups)) + ') will be greater than nb_groups.')

        nb_groups = len(groups)
        nb_nodes = len(set(np.ravel(edges)))
        colored_vertices = ['*'] * nb_nodes

        for i, g in enumerate(groups):

            paulis = shuffle(paulis)

            for j, v in enumerate(g):
                colored_vertices[v] = paulis[j] + ' ' + str(i)

        nb_qubits = nb_groups

    else:

        nb_qubits = int(np.max([int(cv[1:]) for cv in colored_vertices]) + 1)

    if mode in ['11', '21', '31']:

        hamiltonian_strings = []

        for (p, q), w in zip(edges, edge_weights):
            if (p>=len(colored_vertices))or(q>=len(colored_vertices)):
                print(edges)
            i = int(colored_vertices[p][1:])
            j = int(colored_vertices[q][1:])
            si = colored_vertices[p][0]
            sj = colored_vertices[q][0]
            string = 'I' * nb_qubits
            string = string[:i] + si + string[i+1:]
            string = string[:j] + sj + string[j+1:]
            string = string[::-1] # X0Y1 = YX
            hamiltonian_strings += [(string, .5 * len(paulis) * w)]

        hamiltonian_strings += [('I' * nb_qubits, -.5 * np.sum(edge_weights))]

        
    return nb_qubits, colored_vertices, hamiltonian_strings