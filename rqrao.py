from typing import Tuple
from typing import List
from typing import Optional
from typing import Dict
from typing import Union
import numpy as np
import time
import copy
import torch

from qrao import get_qrac_hamiltonian
from mps import get_sparse_hamiltonian_mpo
from mps import get_expectation_sparse
from mps import MPSGradTrainer
from rqaoa import get_reduced_candits
from rqaoa import get_new_graph
from rqaoa import get_best_cut
from rqaoa import get_bits_list


class RQRAO(object):
    
    def __init__(
        self,
        edges: Tuple[Tuple[int, int]],
        edge_weights: Tuple[float],
        bond_dim: int,
        device: str,
        edge_energy_thresh: Optional[float] = None,
        edge_noise: float = 0.,
        thresh: float = 1e-9,
        mode: str = '31',
        brute_force_threshold: int = 10,
        nb_ensemble: int = 1,
        sigma: float = 0.,
        ldf: bool = True,
        optimizer: str = 'lbfgs',
        nb_groups: Optional[int] = None,
    ):
        
        self.edges = edges
        self.edge_weights = edge_weights
        self.bond_dim = bond_dim
        self.device = device
        self.edge_energy_thresh = edge_energy_thresh
        self.edge_noise = edge_noise
        self.thresh = thresh
        self.mode = mode
        self.brute_force_threshold = brute_force_threshold
        self.nb_ensemble = nb_ensemble
        self.sigma = sigma
        self.ldf = ldf
        self.nb_groups = nb_groups
        self.time_dict = {'fit': 0., 'make_mpo': 0., 'edge_energy': 0., 'graph_modification': 0.}
        
        return
    
    
    def get_rqrao_result(self) -> Tuple[float, List[str]]:

        nb_nodes = len(set(np.ravel(self.edges)))

        res_rqrao = self.run_rqrao()

        start = time.time()
        
        cut_max, gs_states = get_best_cut(
            nb_nodes=nb_nodes,
            edges=self.edges,
            edge_weights=self.edge_weights,
            map_list=res_rqrao['map_list'],
            candidate_list=res_rqrao['candidate_list'],
            sgn_edge_energy_list=res_rqrao['sgn_edge_energy_list'],
        )
        
        end = time.time()
        self.time_dict.update({'brute_force': end - start})

        return cut_max, gs_states, self.time_dict
    
    
    def run_rqrao(self) -> Dict[str, Union[Tuple[Tuple[int]], Tuple[Tuple[int, int]], Tuple[int]]]:

        noisy_edge_weights = tuple(np.array(self.edge_weights) + self.edge_noise * (2. * np.random.random(size=len(self.edge_weights)) - 1.))

        edges_new = copy.deepcopy(self.edges)
        edge_weights_new = copy.deepcopy(noisy_edge_weights)
        
        map_list = []
        candidate_list = []
        sgn_edge_energy_list = []

        while True:

            nb_nodes = len(set(np.ravel(edges_new)))

            if nb_nodes <= self.brute_force_threshold:
                break

            edge_energy_dict = self.get_candidate(
                edges=edges_new,
                edge_weights=edge_weights_new,
            )

            candits, parities = get_reduced_candits(
                edge_energy_dict=edge_energy_dict,
                edge_energy_thresh=self.edge_energy_thresh,
            )
            
            start = time.time()

            nb_candits = len(candits)
            
            candidate_list_temp = []
            map_list_temp = []
            sgn_edge_energy_list_temp = []

            for idx in range(nb_candits):

                candidate = candits[idx]
                sgn_edge_energy = parities[idx]

                if frozenset({*candidate}) in [frozenset({*e}) for e in edges_new]:

                    edges_new, edge_weights_new, node_idx_map = get_new_graph(
                        edges=edges_new,
                        edge_weights=edge_weights_new,
                        candidate=candidate,
                        sgn_edge_energy=sgn_edge_energy,
                    )

                    sgn_edge_energy_list_temp += [sgn_edge_energy]
                    candidate_list_temp += [candidate]
                    map_list_temp += [node_idx_map]

                    for i, n in enumerate(node_idx_map):
                        if i != n:
                            org = n
                            replaced = i
                            for j, e in enumerate(candits):
                                if org in e:
                                    if e[0] == org:
                                        candits[j] = (replaced, e[1])
                                    elif e[1] == org:
                                        candits[j] = (e[0], replaced)
                                    break
                                    
                    for j, e in enumerate(candits):
                        if len(node_idx_map) in e:
                            replaced = candidate[0]
                            if e[0] == len(node_idx_map):
                                candits[j] = (replaced, e[1])
                            elif e[1] == len(node_idx_map):
                                candits[j] = (e[0], replaced)
                                
            sgn_edge_energy_list += sgn_edge_energy_list_temp
            candidate_list += candidate_list_temp
            map_list += map_list_temp
            
            end = time.time()
            self.time_dict.update({'graph_modification': self.time_dict['graph_modification'] + end - start})

            if len(edges_new) <= 1:
                break
                
        return {
            'map_list': tuple([tuple(m) for m in map_list]),
            'candidate_list': tuple(candidate_list),
            'sgn_edge_energy_list': tuple(sgn_edge_energy_list),
        }
    

    def get_hamiltonian(
        self,
        edges: Tuple[Tuple[int, int]],
        edge_weights: Tuple[float],
    ) -> Tuple[int, Tuple[str], List[Tuple[str, float]]]:
        
        if self.mode == '11':
            self.paulis = random.sample(['X', 'Y', 'Z'], k=1)
        elif self.mode == '21':
            self.paulis = random.sample(['X', 'Y', 'Z'], k=2)
        elif self.mode == '31':
            self.paulis = 'XYZ'
        else:
            raise

        nb_qubits, colored_vertices, hamiltonian_strings = get_qrac_hamiltonian(
            edges=edges,
            edge_weights=edge_weights,
            mode=self.mode,
            paulis=self.paulis,
            ldf=self.ldf,
            nb_groups=self.nb_groups,
        )

        return nb_qubits, tuple(colored_vertices), hamiltonian_strings
    
    
    def get_candidate(
        self,
        edges: Tuple[Tuple[int, int]],
        edge_weights: Tuple[float],
    ) -> Dict[Tuple[int, int], float]:

        edge_energy_dict = {}

        for ensemble in range(self.nb_ensemble):
            
            exp = self.compute_edge_energies(
                edges=edges,
                edge_weights=edge_weights,
            )

            for i, (p, q) in enumerate(edges):
                if (p, q) in edge_energy_dict.keys():
                    edge_energy_dict.update({(p, q): edge_energy_dict[(p, q)] + [exp[i]]})
                else:
                    edge_energy_dict.update({(p, q): [exp[i]]})

        if self.sigma != 0.:
            
            if len(edges) == 1:
                
                sigma = 0.
                
            else:
                
                e = list(edge_energy_dict.values())
                mean = np.mean(e, 1)
                std = np.std(e, 1)
                indices = np.argsort(np.abs(mean) / std)
                sigma_max = (np.abs(mean) / std)[indices[-2]] + np.finfo(np.float32).eps
                sigma = min(self.sigma, sigma_max)
                
        else:
            
            sigma = self.sigma
            
        edge_energy_dict_ensemble = {}
        for k, v in edge_energy_dict.items():
            if k[0] != k[1]:
                e = np.array(edge_energy_dict[k])
                mean = np.mean(e)
                std = np.std(e)
                e_ensemble = mean - np.sign(mean) * min(sigma * std, abs(mean))
                edge_energy_dict_ensemble.update({k: e_ensemble})
                
        return edge_energy_dict_ensemble
    
    
    def compute_edge_energies(
        self,
        edges: Tuple[Tuple[int, int]],
        edge_weights: Tuple[float],
    ) -> Tuple[float]:
        
        nb_qubits, colored_vertices, hamiltonian_strings = self.get_hamiltonian(edges, edge_weights)

        hamiltonian = []
        hamiltonian_for_edges = []
        for h, v in hamiltonian_strings:
            string = ''
            for i, s in enumerate(reversed(h)):
                if s != 'I':
                    string = string + s + ' ' + str(i) + ' '
            if string != '':
                hamiltonian += [(string[:-1], v)]
                hamiltonian_for_edges += [(string[:-1], 1.)]

        isnan = True
        while isnan:

            mps_init = None

            start = time.time()
            trainer = MPSGradTrainer(
                nb_qubits=nb_qubits,
                bond_dim=self.bond_dim,
                hamiltonian=hamiltonian,
                device=self.device,
                thresh=self.thresh,
                mps_init=mps_init,
            )
            end = time.time()
            self.time_dict.update({'make_mpo': self.time_dict['make_mpo'] + end - start})
            
            start = time.time()
            energy = trainer.fit()
            end = time.time()
            self.time_dict.update({'fit': self.time_dict['fit'] + end - start})

            mps = trainer.model()
            u, v, s = mps
            isnan = torch.sum(torch.stack([torch.isnan(uu).sum() for uu in u])) + torch.sum(torch.stack([torch.isnan(vv).sum() for vv in v])) + torch.sum(torch.isnan(s))

            if isnan:
                print('MPS is nan. Re-initialized.')
                continue
            
        start = time.time()
        
        edge_mpo = get_sparse_hamiltonian_mpo(
            hamiltonian=hamiltonian_for_edges,
            nb_qubits=nb_qubits,
            device=self.device,
            add_header=True,
        )

        exp = tuple(np.real(get_expectation_sparse(edge_mpo, *mps).detach().cpu().numpy()))
        
        end = time.time()
        
        self.time_dict.update({'edge_energy': self.time_dict['edge_energy'] + end - start})
        
        return exp