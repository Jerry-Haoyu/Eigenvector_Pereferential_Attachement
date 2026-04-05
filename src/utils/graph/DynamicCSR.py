"""Class"""

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, csr_matrix
import logging

from utils.graph.AdjMatrix import AdjMatrix

logger = logging.getLogger("DynamicCSR")


class DynamicCSR(AdjMatrix):
    """ Custom sparse hermitian adjacency matrix data structure optimized for fast growing as well as efficient 
        matrix-vector multiplication 
        
        APIs: 
            1. add_vertex_with_edges(v) : adding a vertex, v is the indices of existing vertices the new vertex is 
            going to be connected to 
            2. matvec(v)
            
    """

    def __init__(self, g0: npt.NDArray[np.float64], tau_=150):
        """
        Create an adjacency matrix for size n undirected complete graph with
        new edge buffer to allow growing graph.
        Args: 
            g0: NDArray for initial adjacency matrix of graph
            N: unused
            tau_: buffer cadence/max size of buffer before folding
        """
        
        self.shape : int = (g0.shape)[0] #overall size
        self.csr_shape : int = (g0.shape)[0] #csr size
        self.tau = tau_

        self.A = csr_matrix(g0, dtype=np.int32)

        self.r_buf = np.empty(0, dtype=np.int32)
        self.c_buf = np.empty(0, dtype=np.int32)

    def add_vertex_with_edges(self, v: npt.NDArray[np.integer]):
        """
        Add one new vertex u with undirected edges to 'targets' (int array-like).
        New edges stored in COO-formatted buffer.
        

        @param: v: array of 'targets'/vertex indices the newly added node will connect to
        """
        u = self.shape
        self.shape = self.shape + 1

        # Append both directions (u,t) and (t,u) into the buffer
        self.r_buf = np.concatenate([self.r_buf, np.repeat(u, len(v)), v])
        self.c_buf = np.concatenate([self.c_buf, v, np.repeat(u, len(v))])

        if self.shape - self.csr_shape >= self.tau:
            self.rebuild()

    def rebuild(self):
        """Fold buffer into the main CSR and reset buffer."""
        if self.r_buf.size == 0:
            return

        add = coo_matrix(
            (np.ones(self.r_buf.size, dtype=np.float64), (self.r_buf, self.c_buf)),
            shape=(self.shape, self.shape),
        ).tocsr()

        self.A.resize((self.shape, self.shape))
        self.A = self.A + add

        self.csr_shape = self.shape

        # Clear buffer
        self.r_buf = np.empty(0, dtype=np.int32)
        self.c_buf = np.empty(0, dtype=np.int32)

    def matvec(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """y = (A_main + A_buffer - sigma @ I) @ x, normalized"""
        y = np.zeros(self.shape)
        # Multiply by main CSR
        y[: self.csr_shape] = (self.A @ v[: self.csr_shape]).flatten()

        # handle the buffer multiplication which is just
        # y[i] += A_{ij} * x[j]
        if self.csr_shape < self.shape: 
            np.add.at(
                y, self.r_buf, v[self.c_buf]
            )  # needs add.at since it is repeated idx should be add multiple times

        return y
    
    def get_csr(self) -> csr_matrix:
        """Returns a CSR representation of this matrix"""

        if self.shape != self.csr_shape:
            self.rebuild()

        return self.A