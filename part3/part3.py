import json
import random
import time
import shutil
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset  # added ConcatDataset to merge train+val
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights  # pre-trained model for transfer learning

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

# CHANGED: dataset switched from MNIST to Oxford Flowers102.
# Added num_classes and image_size so the magic numbers aren't scattered through the script.
# Lowered batch_size because 224x224 RGB uses much more memory than 28x28 grayscale.
config = {
    "run_name": "cnn_assignment_part3",
    "dataset": "Flowers102",
    "num_classes": 102,
    "image_size": 224,
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 1e-3,
    "transfer_learning_rate": 1e-4,  # ADDED: lower LR is standard for fine-tuning pre-trained weights
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

# CHANGED: transforms rewritten for 224x224 RGB input.
# - Resize((224, 224)) is required because Flowers102 images come in varying sizes and ResNet50 expects 224.
# - RandomHorizontalFlip + ColorJitter added (flowers look fine mirrored, and color variation helps generalization).
# - Normalize uses ImageNet statistics because that's what ResNet50 was pre-trained with.
#   Using the same normalization for the custom CNNs keeps the comparison fair.
train_transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(saturation=0.2, brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# CHANGED: test transform - no augmentation, only deterministic resize + normalize.
test_transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# CHANGED: dataset loading replaced with Flowers102.
# Quirk of Flowers102: the "train" split is only 1020 images, "val" is 1020, "test" is 6149.
# We merge train+val with ConcatDataset to get 2040 training images and use test for evaluation.
# This is a very common pattern for this dataset.
flowers_train = datasets.Flowers102(root="data", split="train", download=True, transform=train_transform)
flowers_val = datasets.Flowers102(root="data", split="val", download=True, transform=train_transform)
train_dataset = ConcatDataset([flowers_train, flowers_val])
test_dataset = datasets.Flowers102(root="data", split="test", download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

class CNN2Conv(nn.Module):
    # CHANGED: now accepts num_classes so the head is not hardcoded to 10.
    def __init__(self, dropout=0.25, num_classes=102):
        super().__init__()
        # CHANGED: in_channels=3 for RGB (was 1 for MNIST grayscale).
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ADDED: adaptive pooling makes the flattened feature size independent of input resolution.
        # Without this, going from 28x28 to 224x224 would blow up fc1 to ~25M parameters.
        # With adaptive pooling to 7x7, fc1 stays at 64*7*7 = 3136 features, same as the MNIST version.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # CHANGED: output dimension now uses num_classes instead of hardcoded 10.
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # CHANGED: input shape is now (batch, 3, 224, 224) instead of (batch, 1, 28, 28).
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # -> (batch, 32, 112, 112)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # -> (batch, 64, 56, 56)
        x = self.adaptive_pool(x)                        # -> (batch, 64, 7, 7)  ADDED
        x = torch.flatten(x, start_dim=1)                # -> (batch, 3136)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                                  # -> (batch, num_classes) raw logits
        return x

class CNN3Conv(nn.Module):
    # CHANGED: now accepts num_classes.
    def __init__(self, dropout=0.25, num_classes=102):
        super().__init__()
        # CHANGED: in_channels=3 for RGB.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ADDED: same adaptive pooling trick as CNN2Conv.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))

        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        # CHANGED: output uses num_classes.
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # CHANGED: input is (batch, 3, 224, 224) now.
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 224 -> 112
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 112 -> 56
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # 56 -> 28
        x = self.adaptive_pool(x)                        # 28 -> 3   ADDED
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ADDED: factory function for the pre-trained ResNet50.
# Loads ImageNet-pretrained weights, then replaces only the final classification layer
# so the output dimension matches our 102 flower classes.
# The rest of the network (features learned from ImageNet) is kept and fine-tuned.
def build_resnet50_transfer(num_classes=102, **_unused):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # Replace the 1000-class ImageNet head with a new head for our task.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Note: all parameters remain trainable (full fine-tuning).
    # To do pure feature extraction instead, freeze the backbone by setting
    # requires_grad=False on all params BEFORE replacing model.fc.
    return model

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

# CHANGED: architectures list now includes ResNet50 transfer learning as a third entry.
# Each entry is (name, builder_callable, learning_rate) so ResNet50 can use its lower LR.
architectures = [
    ("cnn_2conv", CNN2Conv, config["learning_rate"]),
    ("cnn_3conv", CNN3Conv, config["learning_rate"]),
    ("resnet50_transfer", build_resnet50_transfer, config["transfer_learning_rate"]),  # ADDED
]

hyperparameter_configs = [
    {"run_name": "hp_run_1", "learning_rate": 1e-3, "dropout": 0.25, "weight_decay": 0.0},
    {"run_name": "hp_run_2", "learning_rate": 5e-4, "dropout": 0.25, "weight_decay": 0.0},
    {"run_name": "hp_run_3", "learning_rate": 1e-3, "dropout": 0.40, "weight_decay": 1e-4},
]

if __name__ == "__main__":
    comparison_results = []

    # CHANGED: loop now unpacks three values (name, builder, lr) instead of two.
    # Builder is called with num_classes + dropout so the same code path works for
    # both custom CNNs and the ResNet50 factory.
    for model_name, model_builder, model_lr in architectures:
        print(f"\n===== Training {model_name} =====")

        model = model_builder(dropout=config["dropout"], num_classes=config["num_classes"]).to(DEVICE)
        print(model)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=model_lr,  # CHANGED: per-model LR (ResNet50 gets 1e-4, custom CNNs get 1e-3)
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
            "learning_rate": model_lr,  # ADDED: record which LR was used
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

    # Hyperparameter tuning section — unchanged in structure, only uses CNN2Conv with num_classes now.
    hyperparameter_results = []
    tuning_model_name = "cnn_2conv"

    print("\n=== Hyperparameter tuning ===")
    for hp_config in hyperparameter_configs:
        print(f"\n===== Training {tuning_model_name} with {hp_config} =====")

        # CHANGED: pass num_classes to the constructor.
        model = CNN2Conv(dropout=hp_config["dropout"], num_classes=config["num_classes"]).to(DEVICE)
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

            print(
                f"{hp_config['run_name']} | Epoch {epoch:02d}/{config['epochs']} | "
                f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f}"
            )

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), hp_checkpoint_dir / "best.pt")

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