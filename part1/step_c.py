"""Step C"""
import torch
import torch.nn as nn
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


"""MODEL DEFINITION"""
class OCRPerceptron(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(784, 128) # 784 insputs > 128 neurons
        self.activation1 = nn.Sigmoid() # Activation after layer1
        self.layer2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784) # Reshape: flatten 28x28 image into a 784-length vector

        x = self.layer1(x) # Matrix multiply + bias
        x = self.activation1(x) # Apply Sigmoid element-wise
        x = self.layer2(x) # Second linear layer → raw scores for each digit

        return x


"""DATA LOADING"""
# Skapa en transform för att redan vid load...
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize grey scale
])

# Download MNIST training set (60,000 images of handwritten digits)
train_set = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
# Download MNIST test set (10,000 images)
test_set = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# DataLoader wraps a dataset and provides batches
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

"""SETUP"""
model = OCRPerceptron() # Create an instance of your network
criterion = nn.CrossEntropyLoss() # Loss function for classification (measures how wrong predictions are)

# Adam optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

"""TRAINING LOOP"""
epochs = 5 # Number of full passes through the training data

for epoch in range(epochs):
    model.train()

    # Loop through batches of 64 images + their correct digits
    for images, labels in train_loader: 
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    """EVALUATION"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Epoch {epoch+1}/{epochs}, Accuracy: {correct/total:.4f}")