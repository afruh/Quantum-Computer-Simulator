import numpy as np
import matplotlib.pyplot as plt

from tools import Circuit, gate_dic

class MPSState:
    """class containing the MPS representation of a quantum state"""
    
    def __init__(self, nqbits, bond_dim=2):
        """initialize to |0, 0, ...0>
        Args:
            nqbits (int): the number of qubits
            bond_dim (int): the bond dimension (chi)
        """
        self.nqbits = nqbits
        self.tensors = [np.zeros((1, 2, 1), np.complex128) for _ in range(nqbits)]
        # state |0, 0, ..., 0>
        for qb in range(nqbits):
            self.tensors[qb][0, 0, 0] = 1.0
        self.bond_dim = bond_dim
        
    def apply(self, gate, qbits, param=None):
        """gate is (2, 2) or (4, 4)"""
        # ...
        if len(qbits) > 2: raise Exception(f"The gate acts on too many ({len(qbits)}) qubits")
        gate_ = gate_dic[gate] if param is None else gate_dic[gate](param)
        if len(qbits) == 1:
            # apply the gate
            tmp = np.tensordot(gate_, self.tensors[qbits[0]], axes=[1, 1])
            tmp = np.transpose(tmp, (1, 0, 2))
            self.tensors[qbits[0]] = tmp
        else:
            # put qubits in right order
            if qbits[0] < qbits[1]:
                q1, q2 = qbits[0], qbits[1] 
            else:
                q1, q2 = qbits[1], qbits[0]
                gate_ = np.reshape(gate_, (2, 2, 2, 2))
                # swap q1 and q2
                gate_ = np.transpose(gate_, (1, 0, 3, 2))
                gate_ = np.reshape(gate_, (4, 4))
                
            if q1 != q2 - 1: 
                raise Exception("the gate must be applied on neighboring qubits")
                
            # merge the two neighboring tensors
            tmp = np.tensordot(self.tensors[q1], self.tensors[q2], axes=1)
            lshape, rshape = tmp.shape[0], tmp.shape[3]
            tmp = np.reshape(tmp, (lshape, tmp.shape[1]*tmp.shape[2], rshape))
            # apply the gate
            tmp = np.tensordot(gate_, tmp, axes=[1, 1])
            tmp = np.transpose(tmp, (1, 0, 2))
            
            # now compressing by SVD
            tmp = np.reshape(tmp, (lshape*2, 2*rshape))
            u, s, vh = np.linalg.svd(tmp, full_matrices=False)
            true_bond_dim = min(u.shape[1], self.bond_dim)
            s = s[:true_bond_dim]
            
            # renormalize so that sum s^2 = 1
            s /= np.sqrt(np.sum(s**2)) 
            
            # discard unnecessary columns/rows of u and vh
            u = u[:,:true_bond_dim]
            vh = np.diag(s).dot(vh[:true_bond_dim, :])
            
            self.tensors[q1] = np.reshape(u, (lshape, 2, u.shape[1]))
            self.tensors[q2] = np.reshape(vh, (vh.shape[0], 2, rshape))
            
        
    def __str__(self):
        string = ""
        for qb in range(self.nqbits):
            string += str(self.tensors[qb].shape) + "\n"
        return string
        
        
    def to_vec(self):
        tmp = self.tensors[0]
        for qb in range(1, self.nqbits):
            tmp = np.tensordot(tmp, self.tensors[qb], axes=1)
        return np.reshape(tmp, 2**self.nqbits)
    
class MPSQPU:
    def __init__(self, nqbits, bond_dim=2):
        self.state = MPSState(nqbits, bond_dim)
        
    def submit(self, circuit):
        assert(circuit.nqbits == self.state.nqbits)
        for gate_tuple in circuit.gates:
            self.state.apply(*gate_tuple)
            
        return self.state
    