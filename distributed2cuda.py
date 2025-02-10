from six.moves import urllib
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import time
start_time=time.time()
# Enable dataset download under CloudFlare protection
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert data to torch.FloatTensor
transform = transforms.ToTensor()

# Load the training and test datasets
train_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True,
                             download=True, transform=transform)
test_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=False,
                            download=True, transform=transform)

# Create training and test data loaders
num_workers = 0
batch_size = 20
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# Visualize one image from the dataset
%matplotlib inline
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy()
img = np.squeeze(images[0])
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')

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

# Initialize the model and move it to the device
model = ConvAutoencoder().to(device)
print(model)

# Specify loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Number of epochs to train the model
n_epochs = 30

for epoch in range(1, n_epochs + 1):
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
    print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f}')
end_time=time.time()
print(f"Execution Time After cuda: {end_time-start_time}")