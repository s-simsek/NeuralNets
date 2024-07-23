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
    
    def forward_with_param(
            self, param_name: str, X: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val: np.ndarray) -> np.ndarray:
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward
    

class FullyConnected(Layer):
    
    def __init__(self, n_out:int, activation:str, weight_init="xavier_uniform") -> None:
        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)
        
        # initiate the weights
        self.init_weights = initialize_weights(weight_init, activation=activation)
        
    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters

        Parameters
        ----------
        X_shape : Tuple[int, int]
            Layer shape: (n_in, n_out)
        """
        self.n_in = X_shape[1]
        W = self.init_weights((self.n_in, self.n_out))
        b = np.zeros(shape=(1, self.n_out))
        self.parameters = {'W': W, 'b': b}
        self.cache = {'Z': [], 'X': []}
        self.gradients = {'W': np.zeros_like(W), 'b': np.zeros_like(b)}
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: activation(xW + b)
        Store intermediate results in the cache for backward pass
        
        Parameters
        ----------
        X : np.ndarray
            input matrix of shape (batch_sie, input_dim)

        Returns
        -------
        np.ndarray
            a matrix of shape (batch_size, output_dim)
        """
        
        # if it is the first ever forward call, initialize weights
        if self.n_in is None:
            self._init_parameters(X.shape)
        
        # calculate forward pass
        Z = X @ self.parameters['W'] + self.parameters['b']
        out = self.activation(Z)
        
        # save intermediate results into cache
        self.cache['Z'] = Z
        self.cache['X'] = X
        
        return out 
        
    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        Z = self.cache['Z']
        X = self.cache['X']
        W = self.parameters['W']
        
        # dLdZ = dLdY * dYdZ
        dLdZ = self.activation.backward(Z, dLdY)
        dLdW = X.T @ dLdZ
        dLdB = np.sum(dLdZ, axis=0, keepdims=True)
        dX = dLdZ @ W.T
        
        self.gradients['W'] = dLdW
        self.gradients['b'] = dLdB
        
        return dX
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
