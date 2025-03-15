import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class StepThresholdLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, threshold=1e-6, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        self.threshold = threshold
        super(StepThresholdLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        return [
            max(base_lr * self.gamma ** (self.last_epoch // self.step_size), self.threshold)
            for base_lr in self.base_lrs
        ]
