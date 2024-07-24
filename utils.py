from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.linalg import norm

from NeuralNetworks.optimizers import *

sns.set(style='darkgrid')

def check_gradients(
    fn: Callable[[np.ndarray], np.ndarray],
    grad: np.ndarray,
    x: np.ndarray,
    dLdf: np.ndarray,
    h: float = 1e-6,
) -> float:
    """Performs numerical gradient checking by numerically approximating
    the gradient using a two-sided finite difference.

    For each position in `x`, this function computes the numerical gradient as:
        numgrad = fn(x + h) - fn(x - h)
                  ---------------------
                            2h

    Next, we use the chain rule to compute the derivative of the input of `fn`
    with respect to the loss:
        numgrad = numgrad @ dLdf

    The function then returns the relative difference between the gradients:
        ||numgrad - grad||/||numgrad + grad||

    Parameters
    ----------
    fn       function whose gradients are being computed
    grad     supposed to be the gradient of `fn` at `x`
    x        point around which we want to calculate gradients
    dLdf     derivative of
    h        a small number (used as described above)

    Returns
    -------
    relative difference between the numerical and analytical gradients
    """
    # ONLY WORKS WITH FLOAT VECTORS
    if x.dtype != np.float32 and x.dtype != np.float64:
        raise TypeError(f"`x` must be a float vector but was {x.dtype}")

    # initialize the numerical gradient variable
    numgrad = np.zeros_like(x)

    # compute the numerical gradient for each position in `x`
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        pos = fn(x).copy()
        x[ix] = oldval - h
        neg = fn(x).copy()
        x[ix] = oldval

        # compute the derivative, also apply the chain rule
        numgrad[ix] = np.sum((pos - neg) * dLdf) / (2 * h)
        it.iternext()

    return norm(numgrad - grad) / norm(numgrad + grad)

class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.params = {
            'W1': np.random.randn(input_dim, hidden_dim) * 0.01,
            'b1': np.zeros((1, hidden_dim)),
            'W2': np.random.randn(hidden_dim, output_dim) * 0.01,
            'b2': np.zeros((1, output_dim))
        }
    
    def forward(self, X):
        self.cache = {}
        self.cache['Z1'] = X @ self.params['W1'] + self.params['b1']
        self.cache['A1'] = np.maximum(0, self.cache['Z1'])  # ReLU activation
        self.cache['Z2'] = self.cache['A1'] @ self.params['W2'] + self.params['b2']
        return self.cache['Z2']
    
    def backward(self, X, Y):
        m = X.shape[0]
        dZ2 = self.cache['Z2'] - Y
        dW2 = self.cache['A1'].T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = dZ2 @ self.params['W2'].T
        dZ1 = dA1 * (self.cache['Z1'] > 0)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        return grads

    def update_params(self, grads, optimizer):
        for param_name in self.params.keys():
            delta = optimizer.update(param_name, self.params[param_name], grads['d' + param_name])
            self.params[param_name] -= delta
            
def train_and_plot(optimizer_names, optimizer_params, X_train, y_train, num_epochs=1000):
    input_dim = X_train.shape[1]
    hidden_dim = 10
    output_dim = y_train.shape[1]
    losses = {}

    for optimizer_name in optimizer_names:
        nn = SimpleNN(input_dim, hidden_dim, output_dim)
        optimizer = initialize_optimizer(optimizer_name, **optimizer_params[optimizer_name])
        losses[optimizer_name] = []

        for _ in range(num_epochs):
            logits = nn.forward(X_train)
            loss = np.mean(np.square(logits - y_train))
            losses[optimizer_name].append(loss)

            grads = nn.backward(X_train, y_train)
            nn.update_params(grads, optimizer)
        
     
    num_cols = 2
    num_rows = (len(optimizer_names) + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))
    axs = axs.flatten()
    for i, optimizer_name in enumerate(optimizer_names):
        sns.lineplot(x=range(num_epochs), y=losses[optimizer_name], ax=axs[i])
        axs[i].set_title(f'{optimizer_name} Loss Curve')
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel('Loss')
        
    # Remove any empty subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()
        
