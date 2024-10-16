from typing import Optional
from typing import List
from typing import Tuple
import numpy as np
import time
import torch


class MPSGradTrainer(object):
    
    def __init__(
        self,
        nb_qubits,
        bond_dim,
        hamiltonian,
        device,
        mps_init: Optional[np.ndarray] = None,
        thresh=1e-9,
    ):
        s=time.time()
        self.nb_qubits = nb_qubits
        self.bond_dim = bond_dim
        self.device = device
        self.thresh = thresh
        start = time.time()
        
        nb_hamiltonian_terms = len(hamiltonian)
    
        self.ham_mpo = get_sparse_hamiltonian_mpo(
            hamiltonian=hamiltonian,
            nb_qubits=nb_qubits,
            device=device,
            add_header=False,
        )
            
        self.model = ParametrizedMPS(
            nb_qubits=nb_qubits,
            bond_dim=bond_dim,
            device=device,
            mps_init=mps_init,
        )
                    
        self.hist_loss = []
        
        self.optimizer = self.get_optimizer(self.model, thresh=self.thresh)
        

    def get_optimizer(self, model, lr=1., thresh=1e-9):
        
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lr,
            max_iter=20,
            max_eval=None,
            tolerance_grad=1e-7,
            tolerance_change=thresh,
            history_size=100,
            line_search_fn='strong_wolfe',
        )
        
        return optimizer
    

    def fit(self, mps_normalization=False):
        
        def closure():
            self.optimizer.zero_grad()
            loss = torch.real(get_expectation_sparse(self.ham_mpo, *self.model()))
            loss.backward()
            return loss
        
        itr = 0

        while True:

            loss = self.optimizer.step(closure)
            
            self.hist_loss += [loss.item()]
            
            if np.isnan(self.hist_loss[-1]):
            
                if self.precision == 'double':
                    dtype = torch.float64
                elif self.precision == 'single':
                    dtype = torch.float32
                elif self.precision == 'half':
                    dtype = torch.float16
                else:
                    raise
            
                self.model = ParametrizedMPS(
                    nb_qubits=self.nb_qubits,
                    bond_dim=self.bond_dim,
                    device=self.device,
                    mps_init=None,
                    dtype=dtype,
                )
                self.optimizer = self.get_optimizer(self.model, thresh=self.thresh)
                self.hist_loss = []
                itr = 0
                print('Loss = nan. Re-initialized.')
                
            else:

                if len(self.hist_loss) > 2:


                    if self.hist_loss[-2] - self.hist_loss[-1] < 1e-5:

                        break

                itr += 1
        
        return torch.real(get_expectation_sparse(self.ham_mpo, *self.model())).item()
    
    
def get_sparse_hamiltonian_mpo(
    hamiltonian: List[Tuple[str, float]],
    nb_qubits: int,
    device: str='cpu',
    add_header: bool=False,
):
    
    pauli = {
        'X': np.array([[0., 1.], [1., 0.]]).astype(np.complex128),
        'Y': np.array([[0., -1.j], [1.j, 0.]]).astype(np.complex128),
        'Z': np.array([[1., 0.], [0., -1.]]).astype(np.complex128),
    }
    
    dtype = torch.complex128
    
    mpo_list = []

    eig_0 = [0, 1]
    eig_1 = [1, 0]

    sss=time.time()
    for pos in reversed(range(nb_qubits)):

        mat = []
        idx = []
        val = []

        for s, w in hamiltonian:

            strings = s.split(' ')[0::2]
            positions = [int(p) for p in s.split(' ')[1::2]]

            if pos in positions:

                ss = strings[positions.index(pos)]
                if ss == 'X':
                    idx += eig_1
                    val += [1. + 0.j, 1. + 0.j]
                elif ss == 'Y':
                    idx += eig_1
                    val += [0. - 1.j, 0. + 1.j]
                elif ss == 'Z':
                    idx += eig_0
                    val += [1. + 0.j, -1. + 0.j]
                else:
                    print(s,w)
                    print(ss,pos)
                    raise

                p = pauli[ss]
                mat_i = torch.tensor(p, dtype=dtype, device=device)

            else:

                mat_i = torch.eye(2, device=device)#test
                idx += eig_0
                val += [1. + 0.j, 1. + 0.j]

            if pos == 0:

                mat_i *= w

            mat += [mat_i]

        if pos == 0:

            stacked = torch.stack(mat)
            i, c, d = stacked.shape
            mpo = stacked.reshape(i*c, d).to_sparse()
            
        elif (pos == nb_qubits - 1) and (not add_header):

            stacked = torch.stack(mat)
            j, c, d = stacked.shape
            mpo = stacked.permute(1,2,0).reshape(c*d, j).to_sparse()

        else:

            idx0 = np.stack([np.arange(len(hamiltonian)), np.arange(len(hamiltonian))]).T.ravel()
            idx1 = np.repeat([0, 1], len(hamiltonian)).reshape(2, -1).T.ravel()
            idx2 = np.array(idx)

            merged_idx = []
            for i0, i1, i2 in zip(idx0, idx1, idx2):
                merged_idx += [4*i0 + 2*i1 + i2]
                
            indices = torch.stack([
                torch.tensor(merged_idx, device=device, dtype=torch.long),
                torch.tensor(np.stack([np.arange(len(hamiltonian)), np.arange(len(hamiltonian))]).T.ravel(), device=device, dtype=torch.long)
            ])
            
            val = np.array(val).astype(np.complex128)

            mpo = torch.sparse_coo_tensor(
                indices,
                val,
                (len(hamiltonian)*4, len(hamiltonian)),
                device=device,
            )
            
        mpo_list += [mpo]

        sss=time.time()
        
    return mpo_list


class ParametrizedMPS(torch.nn.Module):
    
    def __init__(
        self,
        nb_qubits: int,
        bond_dim: int,
        device: str,
        mps_init: Optional[np.ndarray] = None,
        dtype: torch.dtype = torch.float64,
    ):
        
        super().__init__()
        
        self.nb_qubits = nb_qubits
        self.bond_dim = bond_dim
        
        if mps_init is None:
            mps_init = get_random_mps(
                nb_qubits=self.nb_qubits,
                bond_dim=bond_dim,
                ndarray=True,
            )

        shapes_u = [m.shape for m in mps_init[0]]
        shapes_v = [m.shape for m in mps_init[1]]
        shapes_lam = [mps_init[2].shape]
        
        matrices_u_real = [torch.nn.Parameter(torch.tensor(np.real(u), dtype=dtype, device=device)) for u in mps_init[0]]
        matrices_u_imag = [torch.nn.Parameter(torch.tensor(np.imag(u), dtype=dtype, device=device)) for u in mps_init[0]]
        matrices_v_real = [torch.nn.Parameter(torch.tensor(np.real(v), dtype=dtype, device=device)) for v in mps_init[1]]
        matrices_v_imag = [torch.nn.Parameter(torch.tensor(np.imag(v), dtype=dtype, device=device)) for v in mps_init[1]]
        matrices_lam_real = [torch.nn.Parameter(torch.tensor(np.real(mps_init[2]), dtype=dtype, device=device))]
        
        self.params_list = torch.nn.ParameterList(matrices_u_real+matrices_u_imag+matrices_v_real+matrices_v_imag+matrices_lam_real)
        self.s_imag = torch.zeros(mps_init[2].shape, dtype=dtype, device=device)


    def forward(self):
        
        vh_real = self.params_list[-3]
        vh_imag = self.params_list[-2]
        s_real = self.params_list[-1]
        vh = torch.complex(vh_real, vh_imag)
        s = torch.complex(s_real, self.s_imag)

        matrices_u = []
        norm = torch.einsum('ia,ja,i,j->ij', vh, torch.conj(vh), s, torch.conj(s))
        
        self.scale = 0
        for i in reversed(range(self.nb_qubits-1, len(self.params_list)-3)):
            u_real = self.params_list[i-self.nb_qubits+1]
            u_imag = self.params_list[i]
            u = torch.complex(u_real, u_imag)
            u2 = u.reshape(u.shape[0]//2,2,u.shape[1])
            norm = torch.einsum('iak,kl->ial', u2, norm)
            norm = torch.einsum('jal,ial->ij', torch.conj(u2), norm)
            if torch.log10(torch.max(torch.real(norm))) > 100.:
                self.scale += 1
                norm = norm / 1e100
            matrices_u += [u]
            
        norm_avg = norm[0, 0]**(.5 / self.nb_qubits) * 1e100**(.5 * self.scale / self.nb_qubits)

        normalized_u = []
        for u in matrices_u[::-1]:
            normalized_u += [u / norm_avg]

        mps = (normalized_u, [vh / norm_avg], s)
        
        return mps
    
    
    def params_normalize(self):        

        norm_avg = 1.
        i = 0
        
        norms = []
        for W in self.parameters():
            norms += [torch.sqrt(torch.sum(W.data * W.data))]
            i += 1
        norm_avg = torch.prod(torch.stack(norms)**(1./float(i)))
        
        for W, n in zip(self.parameters(), norms):
            W.data = W.data / n * norm_avg
            
        return
    
    
def get_random_mps(nb_qubits, bond_dim, device='cpu', ndarray=True):
    
    if type(bond_dim) == int:
        bond_dim = [bond_dim] * nb_qubits
    
    if nb_qubits == 1:
        
        matrices_u = []
        matrices_v = []
        v = np.random.random(size=(2,)) + 1j * np.random.random(size=(2,))
        v = v / np.sqrt(np.sum(v * np.conjugate(v)))
        singular_value = v
        
        return matrices_u, matrices_v, singular_value

    matrices_u = []

    for i in range(nb_qubits-1):

        k = 2 * int(2**(min(min(i, nb_qubits - i), np.log2(bond_dim[i]))))
        l = int(2**(min(min(np.log2(k), nb_qubits - i - 1), np.log2(bond_dim[i]))))
        size = (k, l)
        norm = np.sqrt(l)
        u = np.random.random(size=size) + 1j * np.random.random(size=size)
        u = u / np.sqrt(np.sum(u * np.conjugate(u)))
        matrices_u += [u]

    if bond_dim[-1] == 1:
        size = (1, 2)
    else:
        size = (2, 2)

    v = np.random.random(size=size) + 1j * np.random.random(size=size)
    v = v / np.sqrt(np.sum(v * np.conjugate(v)))
    matrices_v = [v]

    singular_value = np.ones(min(2, bond_dim[-1]))

    norm = np.einsum('ia,ja,i,j->ij', matrices_v[0], np.conjugate(matrices_v[0]), singular_value, np.conjugate(singular_value))
    for u in reversed(matrices_u):
        u2 = u.reshape(u.shape[0]//2, 2, u.shape[1])
        norm = np.einsum('iak,jal,kl->ij', u2, np.conj(u2), norm)
    norm = np.real(norm[0][0])
    
    singular_value /= np.sqrt(norm)

    if ndarray:
        
        return matrices_u, matrices_v, singular_value
    
    else:
        
        matrices_u = [torch.tensor(m, device=device) for m in matrices_u]
        matrices_v = [torch.tensor(m, device=device) for m in matrices_v]
        singular_value = torch.tensor(singular_value, device=device)
        
        return matrices_u, matrices_v, singular_value


def get_expectation_sparse(mpo_sparse, matrices_u, matrices_v, singular_value):

    m = matrices_v[-1]
    o = mpo_sparse[-1]
    ijc, d = o.shape
    c = d = 2
    j = 1
    i = ijc // 2
    b = m.shape[0]
    q = torch.sparse.mm(o, m.T).reshape((i, j, c, b)) # (ijc)d,db->(ijc)b
    Q = torch.einsum('ac,ijcb->iabj', torch.conj(m), q)

    idx = 1

    for m in reversed(matrices_v[:-1]):

        m = m.reshape((m.shape[0], 2, m.shape[1]//2))
        o = mpo_sparse[-1-idx]
        icd, k = o.shape
        c = d = 2
        i = icd // 4
        k, e, f, j = Q.shape
        hQ = torch.sparse.mm(o, Q.reshape((k, e*f*j))).reshape(i, c, d, e, f, j) # (icd)k, k(jef) -> (icd)(jef)
        Q = torch.einsum('icdefj,ace,bdf->iabj', hQ, torch.conj(m), m)

        idx += 1

    Q = torch.einsum('iabj,a,b->iabj', Q, singular_value, singular_value)
    
    for m in reversed(matrices_u):

        m = m.reshape((m.shape[0]//2, 2, m.shape[1]))
        o = mpo_sparse[-1-idx]
        icd, k = o.shape
        c = d = 2
        i = icd // 4
        k, e, f, j = Q.shape
        hQ = torch.sparse.mm(o, Q.reshape((k, e*f*j))).reshape(i, c, d, e, f, j) # (icd)k, k(efj) -> (icd)(efj)
        Q = torch.einsum('icdefj,ace,bdf->iabj', hQ, torch.conj(m), m)

        idx += 1

    exp = torch.ravel(Q)
    
    return exp