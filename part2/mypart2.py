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

print(f"Training samples: {len(train_dataset)}")
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

        # Pooling halves spatial dimensions each time it's applied
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


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)

# Track metrics across epochs
history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
best_test_acc = 0.0

def run_epoch(loader, train_mode):
    if train_mode:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

if __name__ == "__main__":

    training_start = time.time()

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()

        train_loss, train_acc = run_epoch(train_loader, train_mode=True)
        test_loss, test_acc = run_epoch(test_loader, train_mode=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch:02d}/{config['epochs']} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        # Save the "best so far" model. The best model is rarely the last one
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")

        # Save periodic checkpoints every 5 epochs so we can inspect training history.
        if epoch % 5 == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch:02d}.pt")

    total_time = time.time() - training_start
    print(f"\nTotal training time: {total_time:.1f}s")
    print(f"Best test accuracy: {best_test_acc:.4f}")

    with open(run_dir /"history.json", "w") as f:
        json.dump(history, f, indent=2)	

    # Load the best model, not the last epoch's checkpoint
    best_checkpoint_path = checkpoint_dir / "best.pt"
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=DEVICE))

    # Evaluate the test one more time to get the final test loss and accuracy
    final_test_loss, final_test_acc = run_epoch(test_loader, train_mode=False)

    print(f"\n=== Final results === ")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Final test loss: {final_test_loss:.4f}")
    print(f"Final test accuracy: {final_test_acc:.4f}")

    # Save a summary of this run
    summary = {
        "run_name": config["run_name"],
        "run_dir": str(run_dir),
        "config": config,
        "final_test_loss": final_test_loss,
        "final_test_accuracy": final_test_acc,
        "best_test_accuracy": best_test_acc,
        "total_training_time_sec": total_time,
        "epochs_completed": config["epochs"],
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {run_dir / 'summary.json'}")

