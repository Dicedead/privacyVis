from abc import ABC, abstractmethod


class L1Sensitivity(ABC):
    @abstractmethod
    def l1_sens(self) -> float:
        pass

class L2Sensitivity(ABC):
    @abstractmethod
    def l2_sens(self) -> float:
        pass