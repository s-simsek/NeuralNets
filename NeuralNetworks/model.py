from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from NeuralNetworks.layers import FullyConnected
from NeuralNetworks.losses import initialize_loss
from NeuralNetworks.optimizers import initialize_optimizer


class NeuralNetwork:
    def __init__(self, 
                 loss:str, 
                 layer_args: Sequence[Dict],
                 optimizer_args: Dict,
                 ) -> None:
        self.n_layers = len(layer_args)
        self.layer_args = layer_args
        self.loss = initialize_loss(loss)
        self.optimizer = initialize_optimizer(**optimizer_args)
        self._initialize_layers(layer_args)
        
    def _initialize_layers(self, layer_args: Sequence[Dict]) -> None:
        self.layers = []
        for l_arg in layer_args[:-1]: 
            l = FullyConnected(**l_arg)
            self.layers.append(l)
      
    def forward(self, X: np.ndarray) -> np.ndarray:
        temp = X
        for layer in self.layers:
            temp = layer.forward(temp)
        return temp
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        assert y_true.shape == y_pred.shape
        loss = self.loss.forward(y_true, y_pred)
        dLdout = self.loss.backward(y_true, y_pred)
        backward_layers = reversed(self.layers)
        for layer in backward_layers:
            dLdout = layer.backward(dLdout)
        return loss
    
    def update(self) -> None:
        for i, layer in enumerate(self.layers):
            for param_name, param in layer.parameters.items():
                if param_name is not "null":
                    param_grad = layer.gradients[param_name]
                    delta = self.optimizer.update(param_name+str(i), param, param_grad)
                    layer.parameters[param_name] -= delta
            layer.clear_gradients()
    
    def train(self, dataset, epochs) -> None:
        pass
    
    def predict(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, float]:
        pass