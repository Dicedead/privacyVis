import numpy as np

from typing import Callable, Sequence, Any

Constraint = Callable[[np.ndarray, np.ndarray], np.ndarray]
Region = Sequence[Constraint]

TradeOffFunction = Callable[[np.ndarray], np.ndarray] # fn = f(fp)
ScoreFunction = Callable[[np.ndarray, Any], np.ndarray]
