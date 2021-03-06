# SYSTEM IMPORTS
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..parameter import Parameter


class Dense(Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.W: Parameter = Parameter(np.random.randn(in_dim, out_dim))
        self.b: Parameter = Parameter(np.random.randn(1, out_dim))

    # X has shape [num_examples, in_dim]
    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.W.val) + self.b.val

    # because we have learnable parameters here,
    # we need to do 3 things:
    #   1) compute dLoss_dW
    #   2) compute dLoss_db
    #   3) compute (and return) dLoss_dX
    def backwards(self, X: np.ndarray, dLoss_dModule: np.ndarray) -> np.ndarray:
        
        dLoss_dW = np.dot(X.T, dLoss_dModule)
        dLoss_db = np.dot(np.ones([1, X.shape[0]]),dLoss_dModule)
        dLoss_dModule = np.dot(dLoss_dModule, self.W.val.T)
        self.W.grad += dLoss_dW
        self.b.grad += dLoss_db
        return dLoss_dModule

    def parameters(self) -> List[Parameter]:
        return [self.W, self.b]
