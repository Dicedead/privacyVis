import numpy as np

from typing import Callable, Sequence, Any, Tuple

Constraint = Callable[[np.ndarray, np.ndarray], np.ndarray]
Region = Sequence[Constraint]

TradeOffFunction = Callable[[np.ndarray], np.ndarray] # fn = f(fp)
ScoreFunction = Callable[[np.ndarray, Any], np.ndarray]

SLIDER_RESOLUTION_NON_INTEGER = 0.05
SLIDER_RESOLUTION_INTEGER = 1
