#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from models import vae_models
from dataset import VAEDataset

print("[DEBUG] Simple test script starting")

# Load config
with open('configs/minimal_test.yaml', 'r') as file:
    config = yaml.safe_load(file)

print("[DEBUG] Config loaded")

# Create model
model = vae_models[config['model_params']['name']](**config['model_params'])
print("[DEBUG] Model created")

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"[DEBUG] Model moved to {device}")

# Create dataset and set it up
dataset = VAEDataset(**config["data_params"])
print("[DEBUG] Dataset created")

# Set up the dataset (this is required for LightningDataModule)
dataset.setup()
print("[DEBUG] Dataset setup completed")

# Get a single batch using the train_dataloader
dataloader = dataset.train_dataloader()
print("[DEBUG] DataLoader created")

# Get one batch
for batch in dataloader:
    print("[DEBUG] Got batch, shape:", batch.shape)
    break

print("[DEBUG] Simple test completed successfully") 