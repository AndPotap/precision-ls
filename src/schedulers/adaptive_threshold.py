import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class AdaptiveStepLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        step_size,
        gamma=0.1,
        threshold=1e-6,
        last_epoch=-1,
        # For adaptive threshold
        gradient_variance_metric_lambda=0.9,
        gradient_variance_threshold=0.9,
        no_increase_length=1e3,
    ):
        self.step_size = step_size
        self.gamma = gamma
        self.threshold = threshold

        self.gradient_variance_metric_ema = 1.0
        self.gradient_variance_metric_lambda = gradient_variance_metric_lambda
        self.gradient_variance_threshold = gradient_variance_threshold
        self.no_increase_length = no_increase_length
        self.should_decrement = True
        super(AdaptiveStepLR, self).__init__(optimizer, last_epoch)

    def update_gradient_variance_metric(self, gradient_variance_metric):
        self.gradient_variance_metric_ema = self.gradient_variance_metric_lambda * self.gradient_variance_metric_ema + (1 - self.gradient_variance_metric_lambda) * gradient_variance_metric
        if self.gradient_variance_metric_ema < self.gradient_variance_threshold and self.last_epoch > self.no_increase_length:
            self.should_decrement = False
        else:
            self.should_decrement = True

    def get_lr(self):
        new_lrs = []
        for base_lr in self.base_lrs:
            # Start with current learning rate
            lr = self.optimizer.param_groups[0]['lr']

            # Apply multiplicative factor based on the flags
            if self.last_epoch % self.step_size == 0:
                if self.should_decrement:
                    lr = max(lr * self.gamma, self.threshold)  # Decrease LR
                else:
                    lr = lr / self.gamma  # Increase LR

            new_lrs.append(lr)

        return new_lrs
