"""
Part 2 - CNN for MNIST with data augmentation, regularization and basic MLOps.

What this script does, in plain English:
  1. Loads the MNIST handwritten-digit dataset (same data as Part 1).
  2. Applies data augmentation (random rotations, translations, noise) so the
     model sees slightly different versions of each image during training.
  3. Defines a small Convolutional Neural Network (CNN) with batch
     normalization and dropout (two regularization tricks to fight overfitting).
  4. Trains the model, measures accuracy on the test set every epoch and saves
     a checkpoint (a "save file" of the model's weights) whenever the test
     accuracy is the best seen so far. It also saves periodic checkpoints.
  5. Logs metrics to a CSV so each run can be compared later, and writes a
     config.json so you remember exactly which hyperparameters were used.
  6. Runs multiple "experiments" back to back (a tiny hyperparameter search).

Run: python part2.py
"""

import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ----------------------------------------------------------------------------
# 0. Paths & device
# ----------------------------------------------------------------------------
# Reuse the MNIST data already downloaded by Part 1, so we don't download again.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "part1", "data"))
RUNS_DIR = os.path.join(THIS_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

# Use the GPU if one is available, otherwise fall back to the CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------------------------------------------------------------
# 1. Hyperparameter container
# ----------------------------------------------------------------------------
# A dataclass is just a tidy way of grouping related settings in one object.
# Keeping all tunable numbers in one place is a small but important MLOps habit:
# we can dump it to config.json so every run is reproducible.
@dataclass
class Config:
    name: str                 # human-readable tag for the run
    epochs: int = 8
    batch_size: int = 128
    lr: float = 1e-3          # learning rate for Adam
    weight_decay: float = 1e-4  # L2 regularization (fights overfitting)
    dropout: float = 0.25
    conv_channels: tuple = (32, 64)  # channels for each conv layer
    fc_hidden: int = 128      # size of the fully-connected hidden layer
    rotation_deg: float = 10.0
    translate: float = 0.1    # up to 10% horizontal/vertical shift
    noise_std: float = 0.05   # Gaussian noise added to training images
    save_every: int = 2       # save a periodic checkpoint every N epochs


# ----------------------------------------------------------------------------
# 2. Data augmentation + loaders
# ----------------------------------------------------------------------------
def build_loaders(cfg: Config):
    """Build training and test DataLoaders.

    Training pipeline uses augmentation (random rotation, shift, noise) so the
    model learns to recognise digits that are slightly moved or rotated. We do
    *not* augment the test set - that has to stay the true, fixed benchmark.
    """

    class AddGaussianNoise:
        """Tiny custom transform: add random Gaussian noise to a tensor."""
        def __init__(self, std): self.std = std
        def __call__(self, tensor):
            if self.std <= 0:
                return tensor
            return tensor + torch.randn_like(tensor) * self.std

    # MNIST digits are symmetric-ish but NOT mirror-invariant (a "3" flipped
    # horizontally is not a valid digit), so we avoid horizontal flips.
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=cfg.rotation_deg,
            translate=(cfg.translate, cfg.translate),
            scale=(0.9, 1.1),
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
        AddGaussianNoise(cfg.noise_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=train_transform,
    )
    test_set = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=test_transform,
    )

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
    )
    return train_loader, test_loader


# ----------------------------------------------------------------------------
# 3. The CNN model
# ----------------------------------------------------------------------------
class DigitCNN(nn.Module):
    """A small Convolutional Neural Network for 28x28 grayscale images.

    Architecture (why each piece is there):
        Conv2d   -> learns small local features (edges, strokes, curves).
        BatchNorm2d -> keeps activations in a sane range, stabilises training.
        ReLU     -> non-linearity; without it multiple layers collapse into one.
        MaxPool2d -> halves the spatial size, gives translation tolerance.
        (repeat with more channels so later layers can combine simple features
        into more complex ones like "loop" or "intersection")
        Flatten  -> turn the 2D feature maps into a 1D vector.
        Linear + Dropout -> the classifier; dropout randomly zeros activations
                            during training to prevent overfitting.
        Linear   -> final layer with 10 outputs (one score per digit 0..9).
    """

    def __init__(self, conv_channels=(32, 64), fc_hidden=128, dropout=0.25):
        super().__init__()
        c1, c2 = conv_channels

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),  # 1x28x28 -> c1 x28x28
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # -> c1 x14x14

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),  # -> c2 x14x14
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # -> c2 x 7x 7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ----------------------------------------------------------------------------
# 4. Training / evaluation helpers
# ----------------------------------------------------------------------------
def evaluate(model, loader):
    """Return (avg_loss, accuracy) on the given loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / total, correct / total


def train_one_run(cfg: Config):
    """Train a single model end-to-end using `cfg` and save results on disk."""

    # Each run gets its own folder so nothing overwrites anything else.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{cfg.name}"
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save the exact config - crucial for reproducibility.
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # CSV header for per-epoch metrics.
    metrics_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "test_loss", "test_acc", "epoch_seconds"],
        )

    train_loader, test_loader = build_loaders(cfg)

    model = DigitCNN(
        conv_channels=cfg.conv_channels,
        fc_hidden=cfg.fc_hidden,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    # weight_decay in Adam IS L2 regularization - another overfitting fighter.
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    best_acc = 0.0
    run_start = time.time()

    print(f"\n=== Starting run: {run_name} ===")
    print(json.dumps(asdict(cfg), indent=2))

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss, seen = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running_loss / seen
        test_loss, test_acc = evaluate(model, test_loader)
        epoch_secs = time.time() - epoch_start

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f}  test_loss={test_loss:.4f}  "
            f"test_acc={test_acc:.4f}  ({epoch_secs:.1f}s)"
        )

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.6f}", f"{test_loss:.6f}",
                    f"{test_acc:.6f}", f"{epoch_secs:.2f}"],
            )

        # Save the "best so far" model. The best model is rarely the last one
        # because of overfitting, so we keep the one with highest test accuracy.
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": asdict(cfg),
                    "test_acc": test_acc,
                },
                os.path.join(run_dir, "best.pt"),
            )

        # Save periodic checkpoints too so we can inspect training history.
        if epoch % cfg.save_every == 0:
            torch.save(
                model.state_dict(),
                os.path.join(run_dir, f"epoch_{epoch:02d}.pt"),
            )

    total_secs = time.time() - run_start
    print(f"=== Run finished in {total_secs:.1f}s  best_test_acc={best_acc:.4f} ===")

    # Final small summary file for quick scanning.
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(f"run={run_name}\n")
        f.write(f"best_test_acc={best_acc:.4f}\n")
        f.write(f"total_seconds={total_secs:.1f}\n")

    return {"run": run_name, "best_acc": best_acc, "seconds": total_secs}


# ----------------------------------------------------------------------------
# 5. Tiny hyperparameter search
# ----------------------------------------------------------------------------
def main():
    # A hand-written list of experiments. Each one changes ONE or TWO things so
    # the effect of that change is easier to interpret than if everything
    # changed at once. This is the simplest possible hyperparameter tuning.
    experiments = [
        Config(name="baseline"),
        Config(name="bigger_cnn", conv_channels=(32, 64), fc_hidden=256),
        Config(name="more_dropout", dropout=0.4),
        Config(name="less_aug", rotation_deg=0.0, translate=0.0, noise_std=0.0),
    ]

    results = []
    for cfg in experiments:
        results.append(train_one_run(cfg))

    # Print a leaderboard at the end so it's obvious which setup won.
    print("\n====== Leaderboard ======")
    for r in sorted(results, key=lambda x: x["best_acc"], reverse=True):
        print(f"  {r['best_acc']:.4f}   {r['seconds']:6.1f}s   {r['run']}")


if __name__ == "__main__":
    main()
