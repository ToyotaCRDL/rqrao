import numpy as np
from typing import Tuple
from typing import Dict
import time
import subprocess
import os


def get_sdp_result(
    adjacent: np.ndarray,
    matlab_path: str,
    sdpt3_path: str,
) -> Tuple[np.ndarray, Dict[str, float]]:

    time_dict = {}
        
    start = time.time()
    
    cp = subprocess.run('rm ./matrix.txt ./sdpt3_result.txt', shell=True)

    with open('./matrix.txt', 'w') as f:
        for a in adjacent:
            print(','.join([str(w) for w in a.astype(int).tolist()]), file=f)

    end = time.time()
    time_dict.update({'preprocess':end - start})

    #====================================================================================================
    
    with open('./matlab_path.txt', 'w') as f:
        print(matlab_path, file=f)

    with open('./sdpt3_path.txt', 'w') as f:
        print(sdpt3_path, file=f)

    command = matlab_path + ' -nodisplay -nosplash -nodesktop -r "run(' + "'./maxcut_sdpt3.m'" + '); exit;"'
    is_file = os.path.isfile('./sdpt3_result.txt')
    while not is_file:
        cp = subprocess.run(command, shell=True)
        is_file = os.path.isfile('./sdpt3_result.txt')

    with open('./sdpt3_time.txt') as f:
        for line in f:
            sdp_time = float(line)

    time_dict.update({'sdp':sdp_time})

    #====================================================================================================

    start = time.time()

    X = []
    with open('./sdpt3_result.txt') as f:
        for line in f:
            X += [line.split(',')]

    X = np.array(X).astype(np.float64)

    end = time.time()
    time_dict.update({'read X':end - start})

    return X, time_dict


def get_gw_result(
    edges: Tuple[Tuple[int, int]],
    edge_weights: Tuple[float],
    matlab_path: str,
    sdpt3_path: str,
    nb_sampling: int = 1000,
) -> Tuple[float, str, Dict[str, float]]:
    
    nb_nodes = np.max(np.ravel(edges)) + 1
    adjacent = np.zeros((nb_nodes, nb_nodes))
    for (p, q), w in zip(edges, edge_weights):
        adjacent[p, q] = w
        adjacent[q, p] = w
    
    X, time_dict = get_sdp_result(
        adjacent=adjacent,
        matlab_path=matlab_path,
        sdpt3_path=sdpt3_path,
    )
    
    start = time.time()
    L = np.linalg.cholesky(X)
    end = time.time()
    time_dict.update({'cholesky':end - start})

    start = time.time()
    r = np.random.normal(size=(nb_nodes, nb_sampling))
    x = np.sign(L @ r)
    o = np.einsum('ik,ij,jk->k', x, -adjacent, x)
    indices = (np.where(o == np.max(o))[0]).astype(int)
    x_best = (x[:, indices].astype(int) + 1) // 2
    bits_gw = list(set([''.join(x.astype(str))[::-1] for x in x_best.T]))
    end = time.time()
    time_dict.update({'hyperplane cut':end - start})

    start = time.time()
    cut_gw = 0.
    for (p, q), w in zip(edges, edge_weights):
        if bits_gw[0][::-1][p] != bits_gw[0][::-1][q]:
            cut_gw += w
    end = time.time()

    time_dict.update({'compute cut value':end - start})
    end = time.time()
    
    return cut_gw, bits_gw, time_dict


def get_rank2_result(
    edges: Tuple[Tuple[int, int]],
    edge_weights: Tuple[float],
    local_search: bool = True,
    npert: int = 10,
) -> Tuple[float, str]:
    
    dir = os.getcwd() + '/'

    fname = str('graphdata')
    graph_data_file = open(fname, 'wt')
    nb_nodes = int(np.max(list(set(np.ravel(edges))))) + 1
    graph_data_file.write(str(nb_nodes)+' '+str(len(edges))+'\n')
    for (p, q), w in zip(edges, edge_weights):
        graph_data_file.write(str(p+1)+' '+str(q+1)+' '+str(w)+'\n')
    graph_data_file.close()

    if ~os.path.isfile(dir+'param.file'):
        with open(dir+'param.file', 'w') as f:
            print('param.default', file=f)
            print('param', file=f)
            print('--------------------------------------', file=f)
            print('CirCut uses parameters in the 1st file', file=f)

    with open(dir+'param.default', 'w') as f:
        print('max      ................ obj     : max or min', file=f)
        print('cut      ................ task    : cut or bis (bisection)', file=f)
        print('0        ................ plevel  : printout leval: 0, 1 or 2', file=f)
        print('1        ................ init    : initialize t  : 0, 1 or 2', file=f)
        print(str(npert)+'       ................ npert   : No. perturbed restarts >= 0', file=f)
        print('1        ................ multi   : No. of multiple starts >= 1', file=f)
        print('1.e-4    ................ tolf    : tol for relative f-change', file=f)
        print('1.e-4    ................ tolg    : tol for weighted g-norm', file=f)
        print('0.20     ................ pert    : perturbations from a cut', file=f)
        print('0.00     ................ rho     : penalty parameter for bis', file=f)
        print('200      ................ maxiter : maximum iteration number', file=f)
        print('4.0      ................ maxstep : maximum steplength allowed ', file=f)
        if local_search:
            print('T        ................ locsch  : whether local search or not', file=f)
        else:
            print('F        ................ locsch  : whether local search or not', file=f)
        print('T        ................ savecut : save the   cut    x: T or F', file=f)
        print('F        ................ savesol : save the solution t: T or F', file=f)
        print('----------------------------------------------------------------', file=f)
        print('    This file contains the default parameter values.', file=f)
        print('         The above order should not be changed.', file=f)
        
    while True:
        cp = subprocess.run(dir+'circut '+fname, shell=True)
        if cp.returncode == 0:
            break

    with open(os.getcwd()+'/'+fname+'_maxcut.cut') as f:
        dat = [l for l in f]

    os.system('rm '+os.getcwd()+'/'+fname+'_maxcut.cut '+dir+fname)

    cut_rank2 = float([x for x in dat[0].split(' ') if x != ''][2])
    bits_rank2 = ''.join([str((int(x[:-1])+1)//2) for x in dat[1:]])

    return cut_rank2, bits_rank2