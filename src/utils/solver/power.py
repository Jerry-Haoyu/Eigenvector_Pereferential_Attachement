from typing import Callable, Dict, Tuple

import numpy as np
import numpy.typing as npt
from numpy.linalg import norm

def power_iter(
    matvec: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x0: npt.NDArray[np.float64],
    maxiter=500,
    tol=1e-12,
    shift=0.,
    **kwargs
) -> Tuple[np.float64, npt.NDArray[np.float64], Dict]:
    """_summary_

    Args:
        matvec (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]): _description_
        x0 (npt.NDArray[np.float64]): _description_
        maxiter (int, optional): _description_. Defaults to 500.
        tol (_type_, optional): _description_. Defaults to 1e-12.
        shift (_type_, optional): _description_. Defaults to 0..

    Raises:
        RuntimeError: _description_

    Returns:
        Tuple[np.float64, npt.NDArray[np.float64], Dict]: _description_
    """
    success = False
    prev = x0 / norm(x0)
    iter = 0
    oscillation = -1.0
    while iter < maxiter:
        next = matvec(prev) + shift * prev
        next = next / norm(next)
        # print(next)
        # if(iter % 50 == 0):
        #     print(f"at iteration {iter}, the error is {norm(next - prev)}")
        oscillation = norm(next - prev)
        if oscillation < tol:
            success = True
            break
        prev = next
        iter += 1
    if not success:
        raise RuntimeError(f"Eigenvector solve(Power Iteration) failed with oscillation {oscillation} at iteration {maxiter}")
    eigval = np.dot(next, matvec(next))
    return eigval, prev, {"iter": iter, 'oscillation': oscillation}