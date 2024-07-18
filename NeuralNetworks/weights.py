import math
from abc import ABC, abstractmethod

import numpy as np


def initialize_weights(name, activation=None, mode="fan_in"):
    match name:
        case "zeros":
            return Zeros()
        case "ones":
            return Ones()
        case "identity":
            return Identity()
        case "uniform":
            return Uniform()
        case "normal":
            return Normal()
        case "constant":
            return Constant()
        case "sparse":
            return Sparse()
        case "he_uniform":
            return HeUniform(activation=activation, mode=mode)
        case "he_normal":
            return HeNormal(activation=activation, mode=mode)
        case "xavier_uniform":
            return XavierUniform(activation=activation)
        case "xavier_normal":
            return XavierNormal(activation=activation)
        case _:
            raise NotImplementedError


def _calculate_gain(activation, param=None):
    """
    Adapted from https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain
    """
    linear_fns = [
        "linear",
        "conv2d",
    ]
    if (
        activation in linear_fns
        or activation == "sigmoid"
        or activation == "softmax"
    ):
        return 1.0
    elif activation == "tanh":
        return 5.0 / 3.0
    elif activation == "relu":
        return math.sqrt(2.0)
    else:
        return 1.0

def _get_fan(shape, mode="sum"):
    fan_in, fan_out = shape
    if mode == "fan_in":
        return fan_in
    elif mode == "fan_out":
        return fan_out
    elif mode == "sum":
        return fan_in + fan_out
    elif mode == "separate":
        return fan_in, fan_out
    else:
        raise ValueError("Mode must be one of fan_in, fan_out, sum, or separate")

class WeightInitializer(ABC):
    @abstractmethod
    def __call__(self):
        pass

class Zeros(WeightInitializer):
    def __call__(self, shape):
        return np.zeros(shape)

class Ones(WeightInitializer):
    def __call__(self, shape):
        return np.ones(shape)

class Identity(WeightInitializer):
    def __call__(self, shape):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("Identity matrix initializer requires a square 2D shape.")
        return np.eye(shape[0])

class Uniform(WeightInitializer):
    def __call__(self, shape, low=0.0, high=1.0):
        return np.random.uniform(low, high, shape)

class Normal(WeightInitializer):
    def __call__(self, shape, mean=0.0, std=1.0):
        return np.random.normal(mean, std, shape)

class Constant(WeightInitializer):
    def __init__(self, value):
        self.value = value

    def __call__(self, shape):
        return np.full(shape, self.value)

class Sparse(WeightInitializer):
    def __init__(self, sparsity=0.1):
        self.sparsity = sparsity

    def __call__(self, shape):
        array = np.random.randn(*shape)
        mask = np.random.rand(*shape) < self.sparsity
        array *= mask
        return array

class HeUniform(WeightInitializer):
    def __init__(self, activation=None, mode="fan_in"):
        self.activation = activation
        self.mode = mode

    def __call__(self, shape):
        fan = _get_fan(shape, self.mode)
        gain = _calculate_gain(self.activation)
        limit = gain * math.sqrt(6 / fan)
        return np.random.uniform(-limit, limit, shape)

class HeNormal(WeightInitializer):
    def __init__(self, activation=None, mode="fan_in"):
        self.activation = activation
        self.mode = mode

    def __call__(self, shape):
        fan = _get_fan(shape, self.mode)
        gain = _calculate_gain(self.activation)
        std = gain * math.sqrt(2 / fan)
        return np.random.normal(0, std, shape)

class XavierUniform(WeightInitializer):
    def __init__(self, activation=None):
        self.activation = activation

    def __call__(self, shape):
        fan_in, fan_out = _get_fan(shape, mode="separate")
        limit = math.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)

class XavierNormal(WeightInitializer):
    def __init__(self, activation=None):
        self.activation = activation

    def __call__(self, shape):
        fan_in, fan_out = _get_fan(shape, mode="separate")
        std = math.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, std, shape)
