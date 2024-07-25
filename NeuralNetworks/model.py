import math
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
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        
    def _initialize_layers(self, layer_args: Sequence[Dict]) -> None:
        self.layers = []
        for l_arg in layer_args[:-1]: 
            l = FullyConnected(**l_arg)
            self.layers.append(l)
      
    def forward(self, X: np.ndarray) -> np.ndarray:
        """One forward pass through layers

        Parameters
        ----------
        X : np.ndarray
            Data Matrix

        Returns
        -------
        np.ndarray
            predicted labels
        """
        temp = X
        for layer in self.layers:
            temp = layer.forward(temp)
        return temp
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """One backward pass through layers

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            predicted labels

        Returns
        -------
        float
            the total loss
        """
        assert y_true.shape == y_pred.shape
        loss = self.loss.forward(y_true, y_pred)
        dLdout = self.loss.backward(y_true, y_pred)
        backward_layers = reversed(self.layers)
        for layer in backward_layers:
            dLdout = layer.backward(dLdout)
        return loss
    
    def update(self) -> None:
        """Updates the model parameters by chosen optimizer"""
        for i, layer in enumerate(self.layers):
            for param_name, param in layer.parameters.items():
                if param_name is not "null":
                    param_grad = layer.gradients[param_name]
                    delta = self.optimizer.update(param_name+str(i), param, param_grad)
                    layer.parameters[param_name] -= delta
            layer.clear_gradients()
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray, 
              batch_size:int = 32, 
              epochs: int = 500) -> None:
        """Full training cycle

        Parameters
        ----------
        X_train : np.ndarray
            Training data matrix
        y_train : np.ndarray
            Training labels
        batch_size : int, optional
            batch size, by default 32
        epochs : int, optional
            number of epochs to run, by default 500
        """
        
        # save data 
        self.X = X_train
        self.y = y_train 
        
        # initialize output layer
        args = self.layer_args[-1]
        args['n_out'] = len(np.unique(y_train))
        output_layer = FullyConnected(**args) 
        self.layers.append(output_layer)
        
        samples_per_epoch = math.ceil(X_train.shape[0] / batch_size)
        # start training
        for i in range(epochs):
            training_loss = []
            training_error = []
            for iteration in range(samples_per_epoch):
                X, y = self.sample(iteration, batch_size)
                y_pred = self.forward(X)
                loss = self.backward(y, y_pred)
                f1 = self.f1_error(y, y_pred)
                training_loss.append(loss)
                training_error.append(f1)
                self.update()
            print(f'Epoch: {i}, Traning Loss: {np.round(np.mean(training_loss), 2)}, F-1 Score: {np.round(np.mean(training_error), 2)}')       
       
    def sample(self, iteration: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Samples the data for batch processing

        Parameters
        ----------
        iteration : int
            the iteration count in one single epoch
        batch_size : int
            batch size

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Batch X, Batch y
        """
        # shuffle initially
        if iteration == 0:
            idxs = np.arange(self.X.shape[0])
            np.random.shuffle(idxs)
            self.X = self.X[idxs]
            self.y = self.y[idxs]     
        
        low = iteration * batch_size
        high = iteration * batch_size + batch_size
        
        return self.X[low:high], self.y[low:high]
    
    def f1_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the f-1 score

        Parameters
        ----------
        y_true : np.ndarray
            true labels
        y_pred : np.ndarray
            predicted labels

        Returns
        -------
        float
            f-1 score
        """
        # Convert predictions to boolean (0 or 1)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        
        # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        
        # Calculate Precision and Recall
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        # Calculate F1 Score
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            
        return f1
        
        
    def predict(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, float]:
        pass