from abc import ABC

import numpy as np

from NeuralNetworks.schedulers import initialize_scheduler


def initialize_optimizer(
    name,
    lr,
    lr_scheduler=None,
    momentum=None,
    clip_norm=None,
    lr_decay=None,
    staircase=None,
    stage_length=None,
):
    if name == "SGD":
        return SGD(
            lr=lr,
            lr_scheduler=lr_scheduler,
            momentum=momentum,
            clip_norm=clip_norm,
            lr_decay=lr_decay,
            staircase=staircase,
            stage_length=stage_length,
        )
    elif name == "Adam":
        return Adam(
            lr=lr,
            lr_scheduler=lr_scheduler,
            clip_norm=clip_norm,
            lr_decay=lr_decay,
            stage_length=stage_length,
            staircase=staircase,
        )
    elif name == "RMSprop":
        return RMSprop(
            lr=lr,
            lr_scheduler=lr_scheduler,
            clip_norm=clip_norm,
            lr_decay=lr_decay,
            stage_length=stage_length,
            staircase=staircase,
        )
    else:
        raise NotImplementedError


class Optimizer(ABC):
    def __init__(self):
        self.lr = None
        self.lr_scheduler = None


class SGD(Optimizer):
    def __init__(
        self,
        lr,
        lr_scheduler,
        momentum=0.0,
        clip_norm=None,
        lr_decay=0.9,
        stage_length=None,
        staircase=None,
    ):
        self.lr = lr
        self.lr_scheduler = initialize_scheduler(
            lr_scheduler,
            lr=lr,
            decay=lr_decay,
            stage_length=stage_length,
            staircase=staircase,
        )
        self.momentum = momentum
        self.clip_norm = clip_norm
        self.cache = {}

    def update(self, param_name, param, param_grad, epoch):
        if param_name not in self.cache:
            self.cache[param_name] = np.zeros_like(param)

        if self.clip_norm is not None:
            if np.linalg.norm(param_grad) > self.clip_norm:
                param_grad = (
                    param_grad * self.clip_norm / np.linalg.norm(param_grad)
                )

        lr = self.lr_scheduler(epoch)
        delta = (
            self.momentum * self.cache[param_name]
            + lr * param_grad
        )
        self.cache[param_name] = delta
        return delta
    
class Adam(Optimizer):
    def __init__(
        self,
        lr,
        lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        clip_norm=None,
        lr_decay=0.9,
        stage_length=None,
        staircase=None,
    ):
        self.lr = lr
        self.lr_scheduler = initialize_scheduler(
            lr_scheduler,
            lr=lr,
            decay=lr_decay,
            stage_length=stage_length,
            staircase=staircase,
        )
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clip_norm = clip_norm
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, param_name, param, param_grad, epoch):
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)

        if self.clip_norm is not None:
            if np.linalg.norm(param_grad) > self.clip_norm:
                param_grad = (
                    param_grad * self.clip_norm / np.linalg.norm(param_grad)
                )

        self.t += 1
        lr = self.lr_scheduler(epoch)
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * param_grad
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (param_grad ** 2)

        m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)

        delta = lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return delta

class RMSprop(Optimizer):
    def __init__(
        self,
        lr,
        lr_scheduler,
        beta=0.9,
        epsilon=1e-8,
        clip_norm=None,
        lr_decay=0.9,
        stage_length=None,
        staircase=None,
    ):
        self.lr = lr
        self.lr_scheduler = initialize_scheduler(
            lr_scheduler,
            lr=lr,
            decay=lr_decay,
            stage_length=stage_length,
            staircase=staircase,
        )
        self.beta = beta
        self.epsilon = epsilon
        self.clip_norm = clip_norm
        self.cache = {}

    def update(self, param_name, param, param_grad, epoch):
        if param_name not in self.cache:
            self.cache[param_name] = np.zeros_like(param)

        if self.clip_norm is not None:
            if np.linalg.norm(param_grad) > self.clip_norm:
                param_grad = (
                    param_grad * self.clip_norm / np.linalg.norm(param_grad)
                )

        lr = self.lr_scheduler(epoch)
        self.cache[param_name] = self.beta * self.cache[param_name] + (1 - self.beta) * (param_grad ** 2)
        delta = lr * param_grad / (np.sqrt(self.cache[param_name]) + self.epsilon)
        return delta
