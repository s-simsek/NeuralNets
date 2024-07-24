from abc import ABC

import numpy as np


def initialize_optimizer(name, 
                         lr, 
                         momentum=None, 
                         clip_norm=None, 
                         beta1=None, 
                         beta2=None, 
                         epsilon=None):
    match name:
        case "SGD":
            return SGD(lr=lr, 
                    momentum=momentum, 
                    clip_norm=clip_norm)
        case "Adam":
            return Adam(lr=lr, 
                        beta1=beta1, 
                        beta2=beta2, 
                        epsilon=epsilon, 
                        clip_norm=clip_norm)
        case "RMSprop":
            return RMSprop(lr=lr, 
                        beta=beta1, 
                        epsilon=epsilon, 
                        clip_norm=clip_norm)
        
        case "RMSpropNesterov":
            return RMSpropNesterov(lr=lr, 
                                   beta=beta1,
                                   epsilon=epsilon, 
                                   momentum=momentum, 
                                   clip_norm=clip_norm)
        case _:
            raise NotImplementedError
        


class Optimizer(ABC):
    def __init__(self):
        self.lr = None


class SGD(Optimizer):
    def __init__(self, lr, momentum=0.0, clip_norm=None):
        self.lr = lr
        self.momentum = momentum
        self.clip_norm = clip_norm
        self.cache = {}

    def update(self, param_name, param, param_grad):
        if param_name not in self.cache:
            self.cache[param_name] = np.zeros_like(param)

        if self.clip_norm is not None:
            if np.linalg.norm(param_grad) > self.clip_norm:
                param_grad = param_grad * self.clip_norm / np.linalg.norm(param_grad)

        delta = self.momentum * self.cache[param_name] + self.lr * param_grad
        self.cache[param_name] = delta
        return delta


class Adam(Optimizer):
    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8, clip_norm=None):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clip_norm = clip_norm
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, param_name, param, param_grad):
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)

        if self.clip_norm is not None:
            if np.linalg.norm(param_grad) > self.clip_norm:
                param_grad = param_grad * self.clip_norm / np.linalg.norm(param_grad)

        self.t += 1
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * param_grad
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (param_grad ** 2)

        m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)

        delta = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return delta


class RMSprop(Optimizer):
    def __init__(self, lr, beta=0.9, epsilon=1e-8, clip_norm=None):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.clip_norm = clip_norm
        self.cache = {}

    def update(self, param_name, param, param_grad):
        if param_name not in self.cache:
            self.cache[param_name] = np.zeros_like(param)

        if self.clip_norm is not None:
            if np.linalg.norm(param_grad) > self.clip_norm:
                param_grad = param_grad * self.clip_norm / np.linalg.norm(param_grad)

        self.cache[param_name] = self.beta * self.cache[param_name] + (1 - self.beta) * (param_grad ** 2)
        delta = self.lr * param_grad / (np.sqrt(self.cache[param_name]) + self.epsilon)
        return delta
    
class RMSpropNesterov(Optimizer):
    def __init__(self, lr, beta=0.9, epsilon=1e-8, momentum=0.9, clip_norm=None):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.momentum = momentum
        self.clip_norm = clip_norm
        self.cache = {}
        self.velocity = {}

    def update(self, param_name, param, param_grad):
        if param_name not in self.cache:
            self.cache[param_name] = np.zeros_like(param)
        if param_name not in self.velocity:
            self.velocity[param_name] = np.zeros_like(param)

        if self.clip_norm is not None:
            if np.linalg.norm(param_grad) > self.clip_norm:
                param_grad = param_grad * self.clip_norm / np.linalg.norm(param_grad)

        self.cache[param_name] = self.beta * self.cache[param_name] + (1 - self.beta) * (param_grad ** 2)
        
        # Nesterov momentum
        prev_velocity = self.velocity[param_name].copy()
        self.velocity[param_name] = self.momentum * self.velocity[param_name] - self.lr * param_grad / (np.sqrt(self.cache[param_name]) + self.epsilon)
        delta = self.momentum * self.momentum * prev_velocity - (1 + self.momentum) * self.velocity[param_name]

        return delta
