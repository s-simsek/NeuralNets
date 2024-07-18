from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    """Abstract class defining the common interface for all activation methods."""

    def __call__(self, Z):
        return self.forward(Z)

    @abstractmethod
    def forward(self, Z):
        pass
    
    @abstractmethod
    def backward(self, Z, dY):
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
        """Forward pass for f(z) = z.

        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        return Z

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
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
        return dY


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for relu activation:
        f(z) = z if z >= 0
               0 otherwise

        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        return np.maximum(0, Z)

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for relu activation.

        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        # Initialize gradient dZ
        dZ = np.array(dY, copy=True)
        
        # ReLU derivative: dReLU/dZ = 1 if Z > 0, otherwise 0
        dZ[Z <= 0] = 0
        return dZ


class SoftMax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for softmax activation.
        Hint: The naive implementation might not be numerically stable.

        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        Z_stable = Z - np.max(Z, axis=-1, keepdims=True)
        
        # Calculate softmax
        exp_Z = np.exp(Z_stable)
        softmax_output = exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)
        
        return softmax_output

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for softmax activation.

        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
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
        """Forward pass for sigmoid function:
        f(z) = 1 / (1 + exp(-z))

        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        return 1 / (1 + np.exp(-Z))

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for sigmoid.

        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        sigmoid_Z = self.forward(Z)
        dZ = dY * sigmoid_Z * (1 - sigmoid_Z)
        return dZ


class TanH(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for f(z) = tanh(z).

        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        return np.tanh(Z)

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for f(z) = tanh(z).

        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        tanh_Z = self.forward(Z)
        dZ = dY * (1 - np.square(tanh_Z))
        return dZ