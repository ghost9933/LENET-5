import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np

class MNISTDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data, self.targets = torch.load(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]

        # Convert for transforms compatibility
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, label

def compute_mean_std(dataset):
    """
    Compute the mean and standard deviation of a dataset.
    
    Args:
        dataset (Dataset): PyTorch Dataset.
        
    Returns:
        mean (float): Mean of the dataset.
        std (float): Standard deviation of the dataset.
    """
    loader = DataLoader(dataset, batch_size=5000, shuffle=False, num_workers=2, pin_memory=True)
    mean = 0.0
    std = 0.0
    total_samples = 0

    print("Computing mean and standard deviation...")
    for images, _ in loader:
        # images shape: (batch_size, 1, 28, 28)
        batch_samples = images.size(0)
        images = images.view(batch_samples, -1)  # Flatten to (batch_size, 784)
        mean += images.mean(1).sum().item()
        std += images.std(1).sum().item()
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    print(f"Computed Mean: {mean}, Computed Std: {std}")
    return mean, std

def get_data_loaders(data_dir='./', batch_size=64, test_batch_size=1000):
    # Initial transform without normalization to compute mean and std
    initial_transform = transforms.Compose([
        transforms.Pad(2),  
        transforms.ToTensor()
    ])

    # Paths to the .pt files
    train_pt_path = os.path.join(data_dir, 'train', 'MNIST', 'processed', 'training.pt')
    test_pt_path = os.path.join(data_dir, 'test', 'MNIST', 'processed', 'test.pt')

    # Create Dataset instances with initial transform
    train_dataset = MNISTDataset(data_path=train_pt_path, transform=initial_transform)

    # Compute mean and std from the training dataset
    mean, std = compute_mean_std(train_dataset)

    # Define transform with normalization using computed mean and std
    transform = transforms.Compose([
        transforms.Pad(2),  
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    # Update the transform for training and testing datasets
    train_dataset.transform = transform
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
    def imshow(img):
        # Retrieve the mean and std used for normalization
        mean = 0.1307  # Placeholder, will be updated below
        std = 0.3081   # Placeholder, will be updated below

        # To accurately unnormalize, retrieve mean and std from the loader's dataset
        dataset = loader.dataset
        if isinstance(dataset, MNISTDataset):
            transform = dataset.transform
            if isinstance(transform, transforms.Compose):
                # Extract Normalize parameters
                for t in transform.transforms:
                    if isinstance(t, transforms.Normalize):
                        mean = t.mean[0].item()
                        std = t.std[0].item()
                        break

        img = img * std + mean  # Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
        plt.axis('off')  # Hide axis
        plt.show()

    dataiter = iter(loader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images[:num_samples]))
    print('GroundTruth:', ' '.join(f'{classes[j]}' for j in labels[:num_samples]))

if __name__ == "__main__":
    # Example usage
    train_loader, test_loader = get_data_loaders()

    # Define MNIST classes
    classes = [str(i) for i in range(10)]

    # Visualize some training samples
    visualize_samples(train_loader, classes, num_samples=8)
