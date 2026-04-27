import json
import random
import time
import shutil
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
    "run_name": "cnn_assignment_part2",
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
tensorboard_dir = run_dir / "tensorboard"
run_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir.mkdir(parents=True, exist_ok=True)
tensorboard_dir.mkdir(parents=True, exist_ok=True)

# Save a snapshot of the exact training script used for this run
try:
    shutil.copy2(Path(__file__).resolve(), run_dir / "source_snapshot.py")
except Exception:
    pass

# Save config next to the checkpoints
with open(run_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"Device: {DEVICE}")
print(f"Run folder: {run_dir}")

# Data augmentation
# Transforms the training set to add random rotation, translation, and scaling
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
])

# Transforms the test set to normalize the data
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

class CNN2Conv(nn.Module):
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

class CNN3Conv(nn.Module):
    def __init__(self, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 28x28 -> 14x14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 14x14 -> 7x7
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # 7x7 -> 3x3
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

architectures = [
    ("cnn_2conv", CNN2Conv),
    ("cnn_3conv", CNN3Conv),
]

hyperparameter_configs = [
    {"run_name": "hp_run_1", "learning_rate": 1e-3, "dropout": 0.25, "weight_decay": 0.0},
    {"run_name": "hp_run_2", "learning_rate": 5e-4, "dropout": 0.25, "weight_decay": 0.0},
    {"run_name": "hp_run_3", "learning_rate": 1e-3, "dropout": 0.40, "weight_decay": 1e-4},
]

if __name__ == "__main__":
    comparison_results = []

    for model_name, model_class in architectures:
        print(f"\n===== Training {model_name} =====")
    
        model = model_class(dropout=config["dropout"]).to(DEVICE)
        writer = SummaryWriter(log_dir=str(tensorboard_dir / model_name))
        print(model)
        print(f"Trainable parameters: {count_parameters(model)}")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        best_test_acc = 0.0
        training_start = time.time()

        model_checkpoint_dir = checkpoint_dir / model_name
        model_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, config["epochs"] + 1):
            epoch_start = time.time()

            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
            test_loss, test_acc = run_epoch(model, test_loader, criterion)
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/test", test_loss, epoch)
            writer.add_scalar("accuracy/train", train_acc, epoch)
            writer.add_scalar("accuracy/test", test_acc, epoch)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

            epoch_time = time.time() - epoch_start
            print(
                f"{model_name} | Epoch {epoch:02d}/{config['epochs']} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
                f"time={epoch_time:.1f}s"
                )

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), model_checkpoint_dir / "best.pt")

            if epoch % 5 == 0:
                torch.save(model.state_dict(), model_checkpoint_dir / f"epoch_{epoch:02d}.pt")
        writer.flush()
        writer.close()

        total_time = time.time() - training_start

        best_checkpoint_path = model_checkpoint_dir / "best.pt"
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=DEVICE))
        final_test_loss, final_test_acc = run_epoch(model, test_loader, criterion)

        print(f"\n=== Final results for {model_name} ===")
        print(f"Best checkpoint: {best_checkpoint_path}")
        print(f"Final test loss: {final_test_loss:.4f}")
        print(f"Final test accuracy: {final_test_acc:.4f}")

        model_summary = {
            "model_name": model_name,
            "final_test_loss": final_test_loss,
            "final_test_accuracy": final_test_acc,
            "best_test_accuracy": best_test_acc,
            "total_training_time_sec": total_time,
            "epochs_completed": config["epochs"],
        }
        comparison_results.append(model_summary)

        with open(run_dir / f"history_{model_name}.json", "w") as f:
            json.dump(history, f, indent=2)

        with open(run_dir / f"summary_{model_name}.json", "w") as f:
            json.dump(model_summary, f, indent=2)

    with open(run_dir / "architecture_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=2)

    print("\n=== Architecture comparison ===")
    for result in comparison_results:
        print(
            f"{result['model_name']}: "
            f"final_test_accuracy={result['final_test_accuracy']:.4f}, "
            f"training_time={result['total_training_time_sec']:.1f}s"
        )

    hyperparameter_results = []
    tuning_model_name = "cnn_2conv"

    print("\n=== Hyperparameter tuning ===")
    for hp_config in hyperparameter_configs:
        print(f"\n===== Training {tuning_model_name} with {hp_config} =====")

        model = CNN2Conv(dropout=hp_config["dropout"]).to(DEVICE)
        writer = SummaryWriter(log_dir=str(tensorboard_dir / "hyperparameter_tuning" / hp_config["run_name"]))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hp_config["learning_rate"],
            weight_decay=hp_config["weight_decay"],
        )

        best_test_acc = 0.0
        training_start = time.time()

        hp_checkpoint_dir = checkpoint_dir / "hyperparameter_tuning" / hp_config["run_name"]
        hp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, config["epochs"] + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
            test_loss, test_acc = run_epoch(model, test_loader, criterion)
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/test", test_loss, epoch)
            writer.add_scalar("accuracy/train", train_acc, epoch)
            writer.add_scalar("accuracy/test", test_acc, epoch)

            print(
                f"{hp_config['run_name']} | Epoch {epoch:02d}/{config['epochs']} | "
                f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f}"
            )

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), hp_checkpoint_dir / "best.pt")
        writer.flush()
        writer.close()

        total_time = time.time() - training_start

        best_checkpoint_path = hp_checkpoint_dir / "best.pt"
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=DEVICE))
        final_test_loss, final_test_acc = run_epoch(model, test_loader, criterion)

        hp_summary = {
            "model_name": tuning_model_name,
            "run_name": hp_config["run_name"],
            "learning_rate": hp_config["learning_rate"],
            "dropout": hp_config["dropout"],
            "weight_decay": hp_config["weight_decay"],
            "final_test_loss": final_test_loss,
            "final_test_accuracy": final_test_acc,
            "best_test_accuracy": best_test_acc,
            "total_training_time_sec": total_time,
        }
        hyperparameter_results.append(hp_summary)

    with open(run_dir / "hyperparameter_comparison.json", "w") as f:
        json.dump(hyperparameter_results, f, indent=2)
    print(f"Hyperparameter comparison saved to {run_dir / 'hyperparameter_comparison.json'}")
    print(f"TensorBoard logs saved to {tensorboard_dir}")
    print(f"Start TensorBoard with: tensorboard --logdir \"{tensorboard_dir}\"") 