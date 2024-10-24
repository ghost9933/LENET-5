# data.py

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np

class MNISTDataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        Custom Dataset for loading MNIST data from .pt files.

        Args:
            data_path (str): Path to the .pt file (training.pt or test.pt).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data, self.targets = torch.load(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]

        # Convert to PIL Image for transforms compatibility
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(data_dir='./', batch_size=64, test_batch_size=1000):
    """
    Returns training and testing data loaders for the MNIST dataset.

    Args:
        data_dir (str): Base directory containing 'train' and 'test' folders.
        batch_size (int): Batch size for training.
        test_batch_size (int): Batch size for testing.

    Returns:
        train_loader, test_loader: DataLoader objects for training and testing.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Pad(2),  # Pad 28x28 to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Paths to the .pt files
    train_pt_path = './train/MNIST/processed/training.pt'
    test_pt_path = './test/MNIST/processed/test.pt'

    # Create Dataset instances
    train_dataset = MNISTDataset(data_path=train_pt_path, transform=transform)
    test_dataset = MNISTDataset(data_path=test_pt_path, transform=transform)

    # Create DataLoader instances
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, test_loader

def visualize_samples(loader, classes, num_samples=8):
    """
    Visualizes a batch of images from the data loader.

    Args:
        loader (DataLoader): DataLoader object.
        classes (list): List of class names.
        num_samples (int): Number of samples to display.
    """
    def imshow(img):
        img = img * 0.3081 + 0.1307  # Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
        plt.show()

    dataiter = iter(loader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images[:num_samples]))
    print('GroundTruth:', ' '.join(f'{classes[j]}' for j in range(num_samples)))
