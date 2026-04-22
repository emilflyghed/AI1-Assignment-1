import json
import random
import time
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Same seed > same random numbers every run
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Use GPU if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
config = {
    "run_name": "cnn_baseline",
    "dataset": "MNIST",
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "dropout": 0.25,
    "weight_decay": 0.0,
    "seed": SEED,
}

# Creates one folder for each run
timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path("runs") / f"{timestamp}_{config['run_name']}"
checkpoint_dir = run_dir / "checkpoints"
run_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Save config next to the checkpoints
with open(run_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"Device: {DEVICE}")
print(f"Run folder: {run_dir}")

# Transforms the training set to add random rotation, translation, and scaling
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
])

#Transforms the test set to normalize the data
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
])

train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2)

print(f"Traing samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

class CNN(nn.Module):
    def __init__(self, dropout=0.25):
        super().__init__()
        # Convolutional block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Convolutional block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Pooling halves spatioal dimensions each time it's applied
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer
        # After two pools on 28x28 -> 7x7, with 64 channels -> 64*7*7 = 3136 features
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Input shape: (batch, 1, 28, 28)
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # -> (batch, 32, 14, 14)
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # -> (batch, 64, 7, 7)
        x = torch.flatten(x, start_dim=1) # -> (batch, 3136)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # -> (batch, 10) raw logits
        return x

model = CNN(dropout=config["dropout"]).to(DEVICE)
print(model)