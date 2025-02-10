import os
from six.moves import urllib
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import time
start_time=time.time()
# Enable dataset download under CloudFlare protection
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# Initialize distributed process group

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

# Convert data to torch.FloatTensor
transform = transforms.ToTensor()

# Load the training and test datasets
train_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True,
                             download=True, transform=transform)
test_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=False,
                            download=True, transform=transform)

# Define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder layers
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder layers
        self.conv4 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # Compressed representation

        # Decoder
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv4(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.sigmoid(self.conv5(x))

        return x

# Main training script
def main(rank,world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Create data loaders with DistributedSampler if in DDP mode
    train_sampler = DistributedSampler(train_data) if world_size > 1 else None
    test_sampler = DistributedSampler(test_data, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(train_data, batch_size=20, sampler=train_sampler, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=20, sampler=test_sampler, num_workers=0)

    # Initialize model, loss, and optimizer
    model = ConvAutoencoder().to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Number of epochs
    n_epochs = 30

    for epoch in range(1, n_epochs + 1):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        train_loss = 0.0

        # Train the model
        for data in train_loader:
            images, _ = data
            images = images.to(device)  # Move images to the device
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        # Print training statistics
        train_loss /= len(train_loader.dataset)
        if rank == 0:
            print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f}')

    cleanup_ddp()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution Time after data distribution: {execution_time} seconds")
