from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

import numpy as np

from NeuralNetworks.activations import initialize_activation
from NeuralNetworks.weights import initialize_weights


class Layer(ABC):
    
    def __init__(self):
        self.activation = None
        
        self.n_in = None
        self.n_out = None
        
        self.parameters = {}
        self.cache = {}
        self.gradients = {}
        
        super().__init__()
        
    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass
    

class FullyConnected(Layer):
    
    def __init__(self, n_out:int, activation:str, weight_init="xavier_uniform") -> None:
        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)
        
        # initiate the weights
        self.init_weights = initialize_weights(weight_init, activation=activation)
        
    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        self.n_in = X_shape[1]
        # TODO 
