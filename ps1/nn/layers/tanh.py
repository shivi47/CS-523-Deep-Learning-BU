# SYSTEM IMPORTS
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..parameter import Parameter


class Tanh(Module):

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X)

    # no learnable parameters, only need to do one thing:
    #   1) compute (and return) dLoss_dX
    def backwards(self, X: np.ndarray, dLoss_dModule: np.ndarray) -> np.ndarray:
        
        tanhx = self.forward(X)
        dTanh = 1 - np.power(tanhx, 2)
        dLoss_dModule *= dTanh
        return

    def parameters(self) -> List[Parameter]:
        return list()
