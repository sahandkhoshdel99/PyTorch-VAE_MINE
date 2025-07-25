print("[DEBUG] Starting script")
import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.strategies import DDPStrategy

print("[DEBUG] Imports done")

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')
parser.add_argument('--no-wandb', action='store_true',
                    help='disable wandb logging')
parser.add_argument('--test-only', action='store_true',
                    help='test mode - skip training, just test model creation')
parser.add_argument('--no-callbacks', action='store_true',
                    help='disable callbacks to test if they cause memory issues')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

print("[DEBUG] Config loaded")

# Initialize WandbLogger conditionally
if args.no_wandb:
    wandb_logger = None
    print("[DEBUG] WandbLogger disabled")
else:
    wandb_logger = WandbLogger(
        project=config['logging_params'].get('project', 'MINEDisentangleVAE'),
        name=config['logging_params'].get('name', config['model_params']['name']),
        save_dir=config['logging_params']['save_dir'],
        log_model=True
    )
    print("[DEBUG] WandbLogger initialized")

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)
print("[DEBUG] Seed set")

model = vae_models[config['model_params']['name']](**config['model_params'])
print("[DEBUG] Model created")
experiment = VAEXperiment(model,
                          config['exp_params'])
print("[DEBUG] Experiment created")

data = VAEDataset(**config["data_params"], pin_memory=False)
print("[DEBUG] DataModule created")

data.setup()
print("[DEBUG] DataModule setup done")

# Prepare callbacks
callbacks = []
if not args.no_callbacks:
    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(save_top_k=2, 
                        dirpath =os.path.join(config['logging_params']['save_dir'], "checkpoints"), 
                        monitor= "val_loss",
                        save_last= True),
    ]

runner = Trainer(logger=wandb_logger,
                 callbacks=callbacks,
                 # strategy=DDPStrategy(find_unused_parameters=True),  # Temporarily disabled for debugging
                 **config['trainer_params'])
print("[DEBUG] Trainer created")

# Optionally create output directories if needed
Path(os.path.join(config['logging_params']['save_dir'], "Samples")).mkdir(exist_ok=True, parents=True)
Path(os.path.join(config['logging_params']['save_dir'], "Reconstructions")).mkdir(exist_ok=True, parents=True)

print(f"======= Training {config['model_params']['name']} =======")
if args.test_only:
    print("[DEBUG] Test mode - skipping training, just testing model creation")
    print("[DEBUG] Model creation test completed successfully")
else:
    runner.fit(experiment, datamodule=data)
    print("[DEBUG] Training finished")