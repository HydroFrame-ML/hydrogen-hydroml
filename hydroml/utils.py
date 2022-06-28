import torch
import numpy as np
from tqdm.autonotebook import tqdm
from pytorch_lightning import Callback
from pytorch_lightning.callbacks.progress import ProgressBar

def multiply_along_axis(A, B, axis, module=torch):
    return module.swapaxes(module.swapaxes(A, axis, -1) * B, -1, axis)

def divide_along_axis(A, B, axis, module=torch):
    return module.swapaxes(module.swapaxes(A, axis, -1) / B, -1, axis)

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        for k, v in trainer.logged_metrics.items():
            if k not in self.metrics.keys():
                self.metrics[k] = [self._convert(v)]
            else:
                self.metrics[k].append(self._convert(v))

    def _convert(self, x):
        if isinstance(x, torch.Tensor):
            return x.cpu().detach().numpy()
        return x


class LitProgressBar(ProgressBar):
    """
    This just avoids a bug in the progress bar for pytorch lightning
    that causes the progress bar to creep down the notebook
    """
    def init_validation_tqdm(self):
        bar = tqdm(disable=True,)
        return bar
