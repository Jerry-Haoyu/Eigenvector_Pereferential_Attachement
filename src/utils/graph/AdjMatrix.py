"""Definition for base adjacency matrix and wrapper around CSR"""

from abc import ABC, abstractmethod
from typing import TypeVar
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix


class AdjMatrix(ABC):
    """Base class for graph adjacency matrix like types"""

    @abstractmethod
    def __init__(self, g0: npt.NDArray[np.float64], N: np.integer):
        """Initializes with starting graph"""

    @abstractmethod
    def get_csr(self) -> csr_matrix:
        """Returns a CSR representation of this matrix"""

    @abstractmethod
    def matvec(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Performs a matrix vector multiplication"""

    @abstractmethod
    def add_vertex_with_edges(self, v: npt.NDArray[np.integer]):
        """Adds a vertex and connects it with other vertices"""


TAdjMatrix = TypeVar("TAdjMatrix", bound=AdjMatrix)


# class CSRAdjMatrix(AdjMatrix):
#     """Simple wrapper on CSR class"""
#
#     def __init__(self, g0: npt.NDArray[np.float64], N: np.integer):
#         """Initializes with starting graph"""
#         self.g = csr_matrix((N, N))
#         self.n = g0.shape[0]
#         self.g[: self.n, : self.n] = g0
#         self.g_cur = self.g[: self.n, : self.n]
#
#     def get_csr(self) -> csr_matrix:
#         """Returns a CSR representation of this matrix"""
#         return self.g_cur
#
#     def matvec(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
#         """Performs a matrix vector multiplication"""
#         return self.g_cur @ v
#
#     def add_vertex_with_edges(self, v: npt.NDArray[np.integer]):
#         """Adds a vertex and connects it with other vertices"""
#         for vertex in v:
#             self.g[self.n, vertex] = 1
#             self.g[vertex, self.n] = 1
#         self.n += 1
#         self.g_cur = self.g[: self.n, : self.n]


class CSRAdjMatrix(AdjMatrix):
    """Simple wrapper on CSR class"""

    def __init__(self, g0: npt.NDArray[np.float64], N: np.integer):
        """Initializes with starting graph"""
        self.g = csr_matrix(g0)
        self.n = g0.shape[0]

    def get_csr(self) -> csr_matrix:
        """Returns a CSR representation of this matrix"""
        return self.g

    def matvec(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Performs a matrix vector multiplication"""
        return self.g @ v

    def add_vertex_with_edges(self, v: npt.NDArray[np.integer]):
        """Adds a vertex and connects it with other vertices"""
        self.g.resize((self.n + 1, self.n + 1))
        for vertex in v:
            self.g[self.n, vertex] = 1
            self.g[vertex, self.n] = 1
        self.n += 1