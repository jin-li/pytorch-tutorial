import torch
from torchvision import datasets, transforms

# Load the MNIST dataset without any transformations
dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())

# Compute the mean and standard deviation
loader = torch.utils.data.DataLoader(dataset, batch_size=60000, shuffle=False)
data = next(iter(loader))[0]  # Get all the images in a single batch
mean = data.mean().item()
std = data.std().item()

print(f'Mean: {mean}, Std: {std}')