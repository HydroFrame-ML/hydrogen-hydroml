import torch
import mlflow
import numpy as np
import pandas as pd
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

    #def on_train_start(self, trainer, pl_module):
    #    print(trainer.optimizers[0].state_dict())

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


def find_best_checkpoint(
    log_dir,
    experiment_name,
    uri_scheme='file:',
    uri_authority='',
):
    tracking_uri = f'{uri_scheme}{uri_authority}{log_dir}'
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    runs = client.list_run_infos(experiment_id)
    # Assumes first run is current and second is the most recently completed
    test_run = runs[1]
    run_id = test_run.run_id
    run_path = f'{log_dir}/{experiment_id}/{run_id}'
    run_dict = mlflow.get_run(run_id).to_dictionary()
    checkpoint_dir = run_dict['data']['params']['checkpoint_dir']
    tracking_metric = 'train_loss'
    metric_df = pd.read_csv(f'{run_path}/metrics/{tracking_metric}',
                            delim_whitespace=True,
                            names=['time', tracking_metric, 'step'],
                            index_col=2)
    epoch_df = pd.read_csv(f'{run_path}/metrics/epoch',
                            delim_whitespace=True,
                            names=['time', 'epoch', 'step'],
                            index_col=2)

    best_step = metric_df[tracking_metric].idxmin()+1
    current_epoch = int(epoch_df.loc[best_step-1]['epoch'])
    checkpoint_file = f'epoch={current_epoch}-step={best_step}.ckpt'
    best_checkpoint = f'{checkpoint_dir}/{checkpoint_file}'
    return best_checkpoint
