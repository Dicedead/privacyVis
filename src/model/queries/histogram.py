from sensitivities import L1Sensitivity
from query import Query


class Histogram(Query, L1Sensitivity):
    def __init__(self, num_bins: int):
        super().__init__(f)

    def l1_sens(self) -> float:
        return 2.