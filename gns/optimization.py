import torch

class ExponentialPlusConstantLR(torch.optim.lr_scheduler._LRScheduler):
    """Decays the learning rate continuously from `lr_start` to `lr_final`,
    with the delta above `lr_final` decaying by `factor` every
    `steps_per_factor` steps."""
    def __init__(self, optimizer, lr_start, lr_final, factor, steps_per_factor,
                 last_epoch=-1, verbose=False):
        self.lr_start = lr_start
        self.lr_final = lr_final
        self.factor = factor
        self.steps_per_factor = steps_per_factor
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        lr = self.lr_final + (self.lr_start - self.lr_final) * self.factor ** ((self._step_count-1) / self.steps_per_factor)
        return [lr for _ in self.base_lrs]
