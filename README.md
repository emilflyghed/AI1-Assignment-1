# AI1 - Assignment 1

A three-part deep learning assignment, building from a hand-rolled neuron up to
transfer learning with ResNet50. Each part lives in its own folder and contains
the Python script(s) that fulfill the requirements for that part.

## Project structure

```
AI1-Assignment-1/
├── part1/                  # From scratch neuron -> NumPy layer -> PyTorch -> GPU
│   ├── step_a.py           # (A) Single neuron, plain Python and NumPy
│   ├── step_b.py           # (B) NumPy ANN layer (matrix forward pass)
│   ├── step_c.py           # (C) PyTorch perceptron trained on MNIST (CPU)
│   ├── step_d.py           # (D) Same model, executed on a CUDA GPU
│   └── instructions.txt
├── part2/                  # CNN, MLOps, data augmentation, hyperparameter tuning
│   ├── part2.py            # Full training pipeline (MNIST)
│   ├── runs/               # Timestamped runs (checkpoints, configs, TensorBoard)
│   ├── personal_notes/     # Personal summary and TensorBoard screenshots
│   └── instructions.txt
├── part3/                  # Custom CNNs vs. transfer learning (ResNet50)
│   ├── part3.py            # Full training pipeline (Flowers102)
│   ├── runs/               # Timestamped runs (checkpoints, configs, TensorBoard)
│   ├── personal_notes/     # Personal summary and accuracy/loss plots
│   └── instructions.txt
├── requirements.txt
├── .gitignore
└── README.md
```

> Note: Part 1 is split into four small scripts (`step_a.py` … `step_d.py`)
> because the assignment itself defines four sub-steps (A–D). Parts 2 and 3
> are each implemented as a single Python script as required.

## Setup

The project is developed and tested with Python 3.10+ and PyTorch with CUDA.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Datasets (MNIST, Flowers102) are downloaded automatically by `torchvision` on
first run into each part's `data/` folder.

## Part 1 - From a single neuron to a GPU-trained perceptron

Implements a digit-classifying perceptron in four progressive steps:

- **Step A** (`step_a.py`): A single neuron implemented twice. `NeuronA1` uses
  only basic arithmetic in a Python `for` loop, `NeuronA2` uses NumPy vector
  multiplication. Both apply a Leaky-ReLU activation.
- **Step B** (`step_b.py`): An `ANN_layer` class that computes the forward
  pass of an entire layer as a single matrix multiplication in NumPy.
- **Step C** (`step_c.py`): A two-layer PyTorch perceptron (`784 → 128 → 10`)
  with `Sigmoid` activation, trained on MNIST using `Adam` and
  `CrossEntropyLoss`, with per-epoch test accuracy.
- **Step D** (`step_d.py`): Same model as Step C, but moved onto a CUDA GPU
  (`cuda:0`) when available, with batches transferred via `.to(device)`.

Run from `part1/`:

```bash
python step_a.py
python step_b.py
python step_c.py
python step_d.py
```

## Part 2 - CNN, MLOps and hyperparameter tuning on MNIST

A full training pipeline for MNIST that introduces convolutional layers,
regularization, data augmentation and basic MLOps practices.

Highlights of `part2/part2.py`:

- **MLOps**: each run is written to a timestamped folder under `runs/` and
  contains a `config.json`, a snapshot of the source script, per-model
  history/summary JSONs, and TensorBoard logs. The best checkpoint
  (`best.pt`) is updated every time test accuracy improves; periodic
  checkpoints (`epoch_05.pt`, `epoch_10.pt`, …) are also saved.
- **Data augmentation**: training images get `RandomRotation(10°)` and
  `RandomAffine` (translation + scale). Test images are only normalized.
- **Architectures**: two CNNs are compared, `CNN2Conv` (two conv blocks) and
  `CNN3Conv` (three conv blocks), both with `BatchNorm`, `MaxPool` and
  `Dropout`.
- **Hyperparameter tuning**: three preset configurations vary
  `learning_rate`, `dropout` and `weight_decay` and their results are saved
  to `hyperparameter_comparison.json`.

Run from `part2/`:

```bash
python part2.py
tensorboard --logdir runs
```

## Part 3 - Classic learning vs. transfer learning on Flowers102

Solves the same problem on a harder, real-world dataset
([Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), 102 classes
of natural images at 224×224) and compares from-scratch CNNs with a
pre-trained ResNet50.

Highlights of `part3/part3.py`:

- **Dataset**: `train + val` splits are concatenated for training; `test`
  split is used for evaluation.
- **Augmentation**: `Resize(224)`, `RandomHorizontalFlip`, `RandomAffine`,
  `ColorJitter` and ImageNet normalization (so the comparison with ResNet50
  is fair).
- **Architectures compared**:
  - `CNN2Conv` and `CNN3Conv`, adapted from Part 2 to RGB input and 102
    output classes (with `AdaptiveAvgPool2d` to keep the FC layer's input
    size fixed regardless of input resolution).
  - `resnet50_transfer`: ResNet50 with `IMAGENET1K_V1` weights and a
    replaced final `fc` layer for 102 classes, trained with a lower
    learning rate (`1e-4`) appropriate for fine-tuning.
- **Same MLOps pipeline as Part 2**: timestamped runs, source snapshot,
  per-model history/summaries, best+periodic checkpoints, TensorBoard
  logging and a hyperparameter sweep.

Run from `part3/`:

```bash
python part3.py
tensorboard --logdir runs
```

### Result summary (from `part3/personal_notes/summary.txt`)

| Model              | Train accuracy | Test accuracy |
|--------------------|----------------|---------------|
| `CNN2Conv`         | ~11.7%         | ~16.0%        |
| `CNN3Conv`         | ~27.0%         | ~26.7%        |
| `resnet50_transfer`| ~99.4%         | ~92.9%        |

The from-scratch CNNs clearly underfit Flowers102, while transfer learning
with ResNet50 dominates - which is exactly the point the assignment is
trying to illustrate.

## Dependencies

Pinned in `requirements.txt`:

- `numpy`, `scipy`, `Pillow`
- `torch`, `torchvision`
- `tensorboard`
