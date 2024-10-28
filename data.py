import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np

class Dataset(Dataset):
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
    loader = DataLoader(dataset, batch_size=5000, shuffle=False, num_workers=2, pin_memory=True)
    mean = 0.0
    std = 0.0
    total_samples = 0
    for images, _ in loader:
        # images shape: (batch_size, 1, 28, 28)
        batch_samples = images.size(0)
        images = images.view(batch_samples, -1)  # Flatten to (batch_size, 784)
        mean += images.mean(1).sum().item()
        std += images.std(1).sum().item()
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean, std

def get_data_loaders(data_dir='./', batch_size=64, test_batch_size=1000):
    initial_transform = transforms.Compose([
        transforms.Pad(2),  
        transforms.ToTensor()
    ])

    train_pt_path = os.path.join(data_dir, 'train', 'MNIST', 'processed', 'training.pt')
    test_pt_path = os.path.join(data_dir, 'test', 'MNIST', 'processed', 'test.pt')

    train_dataset = Dataset(data_path=train_pt_path, transform=initial_transform)

    mean, std = compute_mean_std(train_dataset)

    transform = transforms.Compose([
        transforms.Pad(2),  
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    train_dataset.transform = transform
    test_dataset = Dataset(data_path=test_pt_path, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, test_loader

def visualize_samples(loader, classes, num_samples=8):
    def imshow(img):
        mean = 0
        std = 0  

        dataset = loader.dataset
        if isinstance(dataset, Dataset):
            transform = dataset.transform
            if isinstance(transform, transforms.Compose):
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
    train_loader, test_loader = get_data_loaders()
    classes = [str(i) for i in range(10)]
    visualize_samples(train_loader, classes, num_samples=8)
