import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import time
start = time.time()
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert data to torch.FloatTensor
transform = transforms.ToTensor()

# Load the training and test datasets
train_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True,
                            download=True, transform=transform)
test_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=False,
                           download=True, transform=transform)

# Create training and test dataloaders
num_workers = 0  # Number of subprocesses for data loading
batch_size = 20  # Number of samples per batch

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# Define the NN architecture
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(28 * 28, encoding_dim)
        # Decoder
        self.fc2 = nn.Linear(encoding_dim, 28 * 28)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Encoder layer
        x = torch.sigmoid(self.fc2(x))  # Decoder layer
        return x

# Initialize the NN and move it to the device (GPU if available)
encoding_dim = 32
model = Autoencoder(encoding_dim).to(device)

# Specify loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Number of epochs to train the model
n_epochs = 20

for epoch in range(1, n_epochs + 1):
    train_loss = 0.0

    # Train the model
    for data in train_loader:
        images, _ = data
        # Move images to the device
        images = images.view(images.size(0), -1).to(device)
        
        # Clear the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate the loss
        loss = criterion(outputs, images)
        
        # Backward pass
        loss.backward()
        
        # Optimization step
        optimizer.step()
        
        # Update running training loss
        train_loss += loss.item() * images.size(0)

    # Print average training loss
    train_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f}')

# Display device usage
print(f"Training completed on: {device}")
end=time.time()
print(f"Execution Time After cuda: {end-start}")