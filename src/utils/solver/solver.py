from typing import Any, Protocol, Callable, Tuple, Dict
import numpy as np
import numpy.typing as npt

class EigSolver(Protocol):
    def __call__(
        self,
        matvec: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        x0: npt.NDArray[np.float64],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[np.float64, npt.NDArray[np.float64], Dict]:
        ...