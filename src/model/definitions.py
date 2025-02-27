import numpy as np

from typing import Callable

Constraint = Callable[[np.ndarray, np.ndarray], np.ndarray]
