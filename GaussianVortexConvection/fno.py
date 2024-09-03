# run - torchrun --nproc_per_node=8 fno.py
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from joblib import dump
import csv
from modulus.models.fno import FNO
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.launch.logging import PythonLogger, LaunchLogger, initialize_mlflow
from sklearn.preprocessing import StandardScaler
import shutil
import os
from torch.nn.parallel import DistributedDataParallel
import random

torch.cuda.empty_cache()

class FNOData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@hydra.main(version_base="1.3", config_path=".", config_name="config_fno.yaml")
def gaussian_vortex_trainer(cfg: DictConfig) -> None:
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # Initialize monitoring
    log = PythonLogger(name="wake_fno")
    log.file_logging()
    initialize_mlflow(
        experiment_name=f"wake_FNO",
        experiment_desc=f"training an FNO model for the Flapping Wak",
        run_name=f"wake FNO training",
        run_desc=f"training FNO for Flapping Wake",
        user_name="Sushrut",
        mode="offline",
    )
    LaunchLogger.initialize(use_mlflow=True)  # Modulus launch logger

    # Define model, loss, optimiser, scheduler, data loader
    model = FNO(
        in_channels=cfg.arch.fno.in_channels,
        out_channels=cfg.arch.decoder.out_features,
        decoder_layers=cfg.arch.decoder.layers,
        decoder_layer_size=cfg.arch.decoder.layer_size,
        dimension=cfg.arch.fno.dimension,
        latent_channels=cfg.arch.fno.latent_channels,
        num_fno_layers=cfg.arch.fno.fno_layers,
        num_fno_modes=cfg.arch.fno.fno_modes,
        padding=cfg.arch.fno.padding,
    ).to(dist.device)

    print("Model initialized:", model)

    loss_fun = MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cfg.scheduler.decay_rate**step
    )


    # Load Data
    data = np.load("data.npy")
    data_u = data[:,:,0:1,:,:]
    data_v = data[:,:,1:,:,:]

    # Create a dictionary to store the scalars
    scalars = {'u': StandardScaler(), 'v': StandardScaler()}

    # Reshape data so that each sample is a separate row
    reshaped_data_u = data_u.reshape(-1,1)
    reshaped_data_v = data_v.reshape(-1,1)

    # Fit and transform the data
    data_u_t = scalars['u'].fit_transform(reshaped_data_u)
    data_v_t = scalars['v'].fit_transform(reshaped_data_v)

    # Reshape the data back to its original shape
    data_u = data_u_t.reshape(data_u.shape)
    data_v = data_v_t.reshape(data_v.shape)
    data = np.concatenate((data_u, data_v), axis=2)

    # Save the dictionary of scalars
    dump(scalars, 'scalars.pkl')

    x = data[:,0,:,:,:]
    y = data[:,1,:,:,:]

    # Convert to torch tensors
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()

    print("x.shape:", x.shape)
    print("y.shape:", y.shape)

    train_dataset = FNOData(x, y)
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    ckpt_args = {
        "path": f"./checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_pseudo_epoch = load_checkpoint(device=dist.device, **ckpt_args)

    # Calculate steps per pseudo epoch
    steps_per_pseudo_epoch = ceil(
        cfg.training.pseudo_epoch_sample_size / cfg.training.batch_size
    )
    validation_iters = ceil(cfg.validation.sample_size / cfg.training.batch_size)

    print("steps_per_pseudo_epoch:", steps_per_pseudo_epoch)
    print("validation_iters:", validation_iters)

    log_args = {
        "name_space": "train",
        "num_mini_batch": steps_per_pseudo_epoch,
        "epoch_alert_freq": 1,
    }
    if cfg.training.pseudo_epoch_sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased pseudo_epoch_sample_size to multiple of batch size: {steps_per_pseudo_epoch*cfg.training.batch_size}"
        )
    if cfg.validation.sample_size % cfg.training.batch_size != 0:
        log.warning(
            f"increased validation sample size to multiple of batch size: {validation_iters*cfg.training.batch_size}"
        )

    # Define forward passes for training and inference
    @StaticCaptureTraining(
        model=model, optim=optimizer, logger=log, use_amp=False, use_graphs=False
    )
    def forward_train(invars, target):
        pred = model(invars)
        loss = loss_fun(pred, target)
        return loss

    @StaticCaptureEvaluateNoGrad(
        model=model, logger=log, use_amp=False, use_graphs=False
    )
    def forward_eval(invars):
        return model(invars)

    if loaded_pseudo_epoch == 0:
        log.success("Training started...")
    else:
        log.warning(f"Resuming training from pseudo epoch {loaded_pseudo_epoch + 1}.")

    for pseudo_epoch in range(max(1, loaded_pseudo_epoch + 1), cfg.training.max_pseudo_epochs + 1):
        # Wrap epoch in launch logger for console / MLFlow logs
        with LaunchLogger(**log_args, epoch=pseudo_epoch) as logger:
            for batch_index, batch in zip(range(steps_per_pseudo_epoch), train_loader):
                x_batch, y_batch = batch
                loss = forward_train(x_batch.to(dist.device), y_batch.to(dist.device))
                logger.log_minibatch({"loss": loss.detach()})
                print(f"Batch {batch_index}: loss = {loss.item()}")
            logger.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        # Save checkpoint
        if pseudo_epoch % cfg.training.rec_results_freq == 0:
            save_checkpoint(**ckpt_args, epoch=pseudo_epoch)

        # Validation step (if needed)
        if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
            with LaunchLogger("valid", epoch=pseudo_epoch) as logger:
                total_loss = 0.0
                predictions = []
                with torch.no_grad():
                    for x_batch, y_batch in test_loader:
                        x_batch = x_batch.to(dist.device)
                        pred_batch = model(x_batch)

                        # Separate the channels
                        pred_batch_u = pred_batch[:,0:1,:,:]
                        pred_batch_v = pred_batch[:,1:,:,:]
                        y_batch_u = y_batch[:,0:1,:,:]
                        y_batch_v = y_batch[:,1:,:,:]

                        # Inverse transform the predictions for each channel
                        pred_batch_u_np = pred_batch_u.cpu().numpy().reshape(-1, 1)
                        pred_original_u = scalars['u'].inverse_transform(pred_batch_u_np)
                        pred_original_u = pred_original_u.reshape(pred_batch_u.shape)

                        pred_batch_v_np = pred_batch_v.cpu().numpy().reshape(-1, 1)
                        pred_original_v = scalars['v'].inverse_transform(pred_batch_v_np)
                        pred_original_v = pred_original_v.reshape(pred_batch_v.shape)

                        # Combine the channels back together
                        pred_original = np.concatenate((pred_original_u, pred_original_v), axis=1)

                        # Inverse transform the targets for each channel
                        y_batch_u_np = y_batch_u.cpu().numpy().reshape(-1, 1)
                        y_original_u = scalars['u'].inverse_transform(y_batch_u_np)
                        y_original_u = y_original_u.reshape(y_batch_u.shape)

                        y_batch_v_np = y_batch_v.cpu().numpy().reshape(-1, 1)
                        y_original_v = scalars['v'].inverse_transform(y_batch_v_np)
                        y_original_v = y_original_v.reshape(y_batch_v.shape)

                        # Combine the channels back together
                        y_original = np.concatenate((y_original_u, y_original_v), axis=1)

                        # Log or process the original predictions and ground truth
                        loss_original = loss_fun(torch.tensor(pred_original), torch.tensor(y_original))
                        total_loss += loss_original.item()
                        predictions.append(pred_original)

                logger.log_epoch({"Validation Loss": total_loss / len(test_loader)})

        # Update learning rate
        if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:
            scheduler.step()

    save_checkpoint(**ckpt_args, epoch=cfg.training.max_pseudo_epochs)
    log.success("Training completed *yay*")

if __name__ == "__main__":
    gaussian_vortex_trainer()