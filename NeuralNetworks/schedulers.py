import math
from abc import ABC, abstractmethod

import numpy as np


def initialize_scheduler(name, lr, decay=None, stage_length=None, 
                         staircase=None, step_size=None, T_max=None):
    
    if name == "constant":
        return Constant(lr=lr)
    elif name == "exponential":
        return Exponential(lr=lr, decay=decay, stage_length=stage_length, staircase=None)
    elif name == "step_decay":
        return StepDecay(lr=lr, decay=decay, step_size=step_size)
    elif name == "cosine_annealing":
        return CosineAnnealing(lr=lr, T_max=T_max)
    else:
        raise NotImplementedError("{} scheduler is not implemented".format(name))


class Scheduler(ABC):
    def __call__(self, epoch):
        return self.scheduled_lr(epoch)

    @abstractmethod
    def scheduled_lr(self, epoch=None):
        pass


class Constant(Scheduler):
    def __init__(self, lr=0.01):
        self.lr = lr

    def scheduled_lr(self, epoch):
        return self.lr


class Exponential(Scheduler):
    def __init__(self, lr=0.01, decay=0.9, stage_length=1000, staircase=False):
        self.lr = lr
        self.decay = decay
        self.stage_length = stage_length
        self.staircase = staircase

    def scheduled_lr(self, epoch):
        if self.staircase:
            stage = math.floor(epoch / self.stage_length)
        else:
            stage = epoch / self.stage_length

        return self.lr * self.decay ** stage
    
class StepDecay(Scheduler):
    def __init__(self, lr=0.01, decay=0.5, step_size=100):
        self.lr = lr
        self.decay = decay
        self.step_size = step_size

    def scheduled_lr(self, epoch):
        return self.lr * (self.decay ** (epoch // self.step_size))

class CosineAnnealing(Scheduler):
    def __init__(self, lr=0.01, T_max=100):
        self.lr = lr
        self.T_max = T_max

    def scheduled_lr(self, epoch):
        return self.lr * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
