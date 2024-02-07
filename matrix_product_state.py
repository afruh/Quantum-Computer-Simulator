import numpy as np
from tools import gate_dic

class MPSState:
    def __init__(self, nqbits, chi,):
        self.A = np.zeros((chi,2,chi),dtype=complex)