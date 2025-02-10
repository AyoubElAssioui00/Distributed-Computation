import os
import torch
import torch.distributed as dist
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import time
start = time.time()
def setup(rank, world_size):
    """Setup distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Cleanup distributed process group."""
    dist.destroy_process_group()

class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        self.fc1 = nn.Linear(28 * 28, encoding_dim)
        self.fc2 = nn.Linear(encoding_dim, 28 * 28)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def train_ddp(rank, world_size, encoding_dim, n_epochs):
    setup(rank, world_size)
    
    # Set device for the rank
    device = torch.device(f"cuda:{rank}")
    
    # Transform and dataset
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True,
                                download=True, transform=transform)
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=20, sampler=train_sampler, num_workers=0)
    
    # Model, loss, and optimizer
    model = Autoencoder(encoding_dim).to(device)
    ddp_model = DDP(model, device_ids=[rank])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(1, n_epochs + 1):
        train_sampler.set_epoch(epoch)  # Shuffle data for each epoch
        train_loss = 0.0
        for data in train_loader:
            images, _ = data
            images = images.view(images.size(0), -1).to(device)
            
            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        train_loss /= len(train_loader.dataset)
        print(f"Rank {rank}, Epoch: {epoch}, Training Loss: {train_loss:.6f}")
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # Number of GPUs
    if world_size < 2:
        raise RuntimeError("Requires at least 2 GPUs for DDP")
    
    encoding_dim = 32
    n_epochs = 20
    
    # Launch training on each GPU
    torch.multiprocessing.spawn(train_ddp, args=(world_size, encoding_dim, n_epochs), nprocs=world_size, join=True)
    end=time.time()
    print(f"Execution Time After data distribution: {end-start}")
    