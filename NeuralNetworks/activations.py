from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    """Abstract class defining the common interface for all activation methods."""

    def __call__(self, Z):
        return self.forward(Z)

    @abstractmethod
    def forward(self, Z):
        """Forward pass for f(z) = z.

        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        pass
    
    @abstractmethod
    def backward(self, Z, dY):
        """Backward pass for f(z) = z.

        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        pass

def initialize_activation(name: str) -> Activation:
    """Factory method to return an Activation object of the specified type."""
    match name: 
        case "linear":
            return Linear()
        case "sigmoid":
            return Sigmoid()
        case "tanh":
            return TanH()
        case "relu":
            return ReLU()
        case "softmax":
            return SoftMax()
        case _:
            raise NotImplementedError("{} activation is not implemented".format(name))
    
class Linear(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        return Z

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        return dY


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        # Initialize gradient dZ
        dZ = np.array(dY, copy=True)
        
        # ReLU derivative: dReLU/dZ = 1 if Z > 0, otherwise 0
        dZ[Z <= 0] = 0
        return dZ


class SoftMax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        Z_stable = Z - np.max(Z, axis=-1, keepdims=True)
        
        # Calculate softmax
        exp_Z = np.exp(Z_stable)
        softmax_output = exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)
        
        return softmax_output

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        softmax_output = self.forward(Z)
        
        # Initialize the gradient with the same shape as dY
        dZ = np.zeros_like(Z)
        
        # For each sample in the batch, compute the Jacobian matrix
        for i in range(len(Z)):
            # Flatten the softmax output for the current sample
            s = softmax_output[i].reshape(-1, 1)
            
            # Compute the Jacobian matrix for the softmax function
            jacobian_matrix = np.diagflat(s) - np.dot(s, s.T)
            
            # Compute the gradient for the current sample
            dZ[i] = np.dot(jacobian_matrix, dY[i])
        
        return dZ


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-Z))

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        sigmoid_Z = self.forward(Z)
        dZ = dY * sigmoid_Z * (1 - sigmoid_Z)
        return dZ


class TanH(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        return np.tanh(Z)

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        tanh_Z = self.forward(Z)
        dZ = dY * (1 - np.square(tanh_Z))
        return dZ