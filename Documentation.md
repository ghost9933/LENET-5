# LeNet-5 Implementation with PyTorch: A Comprehensive Tutorial

![LeNet-5 Architecture](https://miro.medium.com/max/1400/1*Ie1wvnNAX7EivocKVRe14g.png)

*Figure 1: Original LeNet-5 Architecture*

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Understanding LeNet-5](#2-understanding-lenet-5)
   - [What is LeNet-5?](#21-what-is-lenet-5)
   - [Architecture Overview](#22-architecture-overview)
   - [Mathematical Foundations](#23-mathematical-foundations)
3. [Project Structure](#3-project-structure)
4. [Module Breakdown](#4-module-breakdown)
   - [1. `data.py`](#41-datapy)
   - [2. `model.py`](#42-modelpy)
   - [3. `train.py`](#43-trainpy)
   - [4. `evaluate.py`](#44-evaluatepy)
   - [5. `utils.py`](#45-utilspy)
   - [6. `main.py`](#46-mainpy)
5. [Detailed Implementation](#5-detailed-implementation)
   - [1. Data Loading and Preprocessing](#51-data-loading-and-preprocessing)
   - [2. Building the LeNet-5 Model](#52-building-the-lenet-5-model)
   - [3. Training the Model](#53-training-the-model)
   - [4. Evaluating the Model](#54-evaluating-the-model)
6. [Running the Project](#6-running-the-project)
7. [Understanding the Components](#7-understanding-the-components)
   - [Convolutional Layers](#71-convolutional-layers)
   - [Activation Functions](#72-activation-functions)
   - [Pooling Layers](#73-pooling-layers)
   - [Fully Connected Layers](#74-fully-connected-layers)
   - [Loss Functions and Optimizers](#75-loss-functions-and-optimizers)
8. [Visualizing Results](#8-visualizing-results)
9. [Conclusion](#9-conclusion)
10. [References](#10-references)

---

## 1. Introduction

Welcome to the comprehensive tutorial on implementing the LeNet-5 Convolutional Neural Network (CNN) using PyTorch. This guide is designed for individuals with a basic understanding of Python who are venturing into the realm of machine learning and deep learning. By the end of this tutorial, you'll not only have a working implementation of LeNet-5 but also a deep understanding of its components and the underlying mathematics.

---

## 2. Understanding LeNet-5

### 2.1. What is LeNet-5?

LeNet-5 is one of the pioneering Convolutional Neural Networks (CNNs) developed by Yann LeCun and his colleagues in 1998. It was primarily designed for handwritten digit recognition, specifically on the MNIST dataset. Despite being over two decades old, LeNet-5 laid the foundational architecture principles that modern CNNs build upon.

### 2.2. Architecture Overview

The LeNet-5 architecture consists of the following layers:

1. **Input Layer**: Receives 32x32 grayscale images.
2. **Convolutional Layer 1 (C1)**: Applies 6 filters of size 5x5, stride 1.
3. **Activation Function**: `tanh`.
4. **Average Pooling Layer 1 (S2)**: 2x2 kernel, stride 2.
5. **Convolutional Layer 2 (C3)**: Applies 16 filters of size 5x5.
6. **Activation Function**: `tanh`.
7. **Average Pooling Layer 2 (S4)**: 2x2 kernel, stride 2.
8. **Flattening**: Converts 2D feature maps to 1D feature vectors.
9. **Fully Connected Layer 1 (C5)**: 120 neurons.
10. **Activation Function**: `tanh`.
11. **Fully Connected Layer 2 (F6)**: 84 neurons.
12. **Activation Function**: `tanh`.
13. **Output Layer**: 10 neurons (for digit classification) with `softmax` activation.

![LeNet-5 Detailed Architecture](https://upload.wikimedia.org/wikipedia/commons/2/26/LeNet-5.svg)

*Figure 2: Detailed LeNet-5 Architecture*

### 2.3. Mathematical Foundations

Understanding the mathematical operations behind each layer is crucial. Let's delve into the core components:

#### 2.3.1. Convolution Operation

The convolutional layer applies a set of learnable filters (kernels) to the input. Each filter convolves across the width and height of the input volume, computing dot products between the entries of the filter and the input at any position.

**Mathematically:**

For an input \( X \) and a filter \( K \), the convolution operation is:

\[
Y(i, j) = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} X(i+m, j+n) \cdot K(m, n)
\]

Where:
- \( Y \) is the output feature map.
- \( M \times N \) is the filter size.

#### 2.3.2. Activation Function (`tanh`)

After convolution, an activation function introduces non-linearity into the model, allowing it to learn complex patterns.

**`tanh` Function:**

\[
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\]

Properties:
- Output range: (-1, 1).
- Zero-centered, which helps in faster convergence during training.

#### 2.3.3. Pooling Operation

Pooling reduces the spatial dimensions (width and height) of the input, which decreases the computational load and helps in making the detection of features invariant to scale and orientation changes.

**Average Pooling:**

Calculates the average of elements within a window.

\[
Y(i, j) = \frac{1}{M \times N} \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} X(i+m, j+n)
\]

#### 2.3.4. Fully Connected Layers

These layers are traditional neural network layers where each neuron is connected to every neuron in the previous layer. They are responsible for high-level reasoning in the network.

#### 2.3.5. Loss Function (`CrossEntropyLoss`)

Measures the performance of the classification model whose output is a probability value between 0 and 1.

\[
L = -\sum_{c=1}^{C} y_c \log(p_c)
\]

Where:
- \( y_c \) is the true label.
- \( p_c \) is the predicted probability.

#### 2.3.6. Optimizer (`SGD` with Momentum)

Stochastic Gradient Descent (SGD) updates the network's weights in the direction that minimizes the loss function.

\[
v_t = \gamma v_{t-1} + \eta \nabla L
\]
\[
\theta = \theta - v_t
\]

Where:
- \( v_t \) is the velocity.
- \( \gamma \) is the momentum coefficient.
- \( \eta \) is the learning rate.
- \( \nabla L \) is the gradient of the loss.

---

## 3. Project Structure

Organizing your project into modules enhances readability, maintainability, and scalability. Below is the recommended directory structure:

```
lenet5_project/
├── data/
│   ├── train/
│   │   └── MNIST/
│   │       ├── processed/
│   │       │   └── training.pt
│   │       └── raw/
│   └── test/
│       └── MNIST/
│           ├── processed/
│           │   └── test.pt
│           └── raw/
├── data.py
├── evaluate.py
├── main.py
├── model.py
├── requirements.txt
├── train.py
└── utils.py
```

**Module Descriptions:**

- **`data.py`**: Handles data loading and preprocessing.
- **`model.py`**: Defines the LeNet-5 architecture.
- **`train.py`**: Contains the training loop.
- **`evaluate.py`**: Includes evaluation metrics and functions.
- **`utils.py`**: Utility functions (e.g., plotting, saving models).
- **`main.py`**: The entry point that ties everything together.
- **`requirements.txt`**: Lists all dependencies.
- **`data/`**: Directory containing the MNIST dataset split into training and testing sets.

---

## 4. Module Breakdown

Let's explore each module in detail.

### 4.1. `data.py`

**Purpose**: Load and preprocess the MNIST dataset from preprocessed `.pt` files.

**Key Components**:
- `MNISTDataset` class: Custom dataset loader.
- `get_data_loaders` function: Returns training and testing data loaders.
- `visualize_samples` function: (Optional) Visualizes sample images.

### 4.2. `model.py`

**Purpose**: Define the LeNet-5 neural network architecture.

**Key Components**:
- `LeNet5` class: Implements the layers and forward pass.

### 4.3. `train.py`

**Purpose**: Handle the training loop, including forward and backward passes.

**Key Components**:
- `train_model` function: Trains the model over multiple epochs.

### 4.4. `evaluate.py`

**Purpose**: Evaluate the trained model's performance.

**Key Components**:
- `evaluate_model` function: Computes overall accuracy.
- `evaluate_per_class` function: Computes per-class accuracy.

### 4.5. `utils.py`

**Purpose**: Provide utility functions for saving/loading models and plotting.

**Key Components**:
- `save_model` function: Saves the model's state.
- `load_model` function: Loads a saved model.
- `plot_training_loss` function: Plots the training loss over epochs.

### 4.6. `main.py`

**Purpose**: Integrate all modules to execute the complete workflow.

**Key Components**:
- Sets hyperparameters.
- Loads data.
- Initializes the model.
- Trains the model.
- Saves the trained model.
- Evaluates the model.
- Plots training loss.

---

## 5. Detailed Implementation

Let's delve into each module with code snippets and explanations.

### 5.1. Data Loading and Preprocessing (`data.py`)

```python
# data.py

import torch
from torch.utils.data import Dataset, DataLoader
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

def get_data_loaders(data_dir='./data', batch_size=64, test_batch_size=1000):
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
    train_pt_path = os.path.join(data_dir, 'train', 'MNIST', 'processed', 'training.pt')
    test_pt_path = os.path.join(data_dir, 'test', 'MNIST', 'processed', 'test.pt')

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
```

**Explanation:**

- **`MNISTDataset` Class**:
  - **Initialization**: Loads the `.pt` file containing MNIST data and labels.
  - **`__len__`**: Returns the total number of samples.
  - **`__getitem__`**: Retrieves the image and label at a given index. Converts the tensor image to a PIL Image to apply transformations.

- **`get_data_loaders` Function**:
  - **Transformations**:
    - **Padding**: Increases image size from 28x28 to 32x32 by adding 2 pixels of padding on each side. This aligns with LeNet-5's input expectations.
    - **ToTensor**: Converts PIL Images to PyTorch tensors.
    - **Normalize**: Standardizes the data using the mean and standard deviation of the MNIST dataset.
  
  - **Data Loaders**:
    - **`train_loader`**: Shuffles the data for training and uses multiple workers for efficient loading.
    - **`test_loader`**: Does not shuffle data, as shuffling isn't necessary during evaluation.

- **`visualize_samples` Function**: (Optional)
  - Displays a grid of sample images along with their ground truth labels to verify data loading.

---

### 5.2. Building the LeNet-5 Model (`model.py`)

```python
# model.py

import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)  # 1x32x32 -> 6x28x28
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 6x28x28 -> 6x14x14
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)  # 6x14x14 -> 16x10x10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 16x10x10 -> 16x5x5

        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # Activation Function
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.pool1(x)
        x = self.tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)  # Flatten
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
```

**Explanation:**

- **Convolutional Layers**:
  - **`conv1`**: 
    - **Input Channels**: 1 (grayscale images).
    - **Output Channels**: 6 filters.
    - **Kernel Size**: 5x5.
    - **Output Size**: 6 feature maps of size 28x28.
  
  - **`pool1`**:
    - **Type**: Average Pooling.
    - **Kernel Size**: 2x2.
    - **Stride**: 2.
    - **Output Size**: Reduces each feature map to 14x14.
  
  - **`conv2`**:
    - **Input Channels**: 6.
    - **Output Channels**: 16 filters.
    - **Kernel Size**: 5x5.
    - **Output Size**: 16 feature maps of size 10x10.
  
  - **`pool2`**:
    - **Type**: Average Pooling.
    - **Kernel Size**: 2x2.
    - **Stride**: 2.
    - **Output Size**: Reduces each feature map to 5x5.

- **Fully Connected Layers**:
  - **`fc1`**: Connects the flattened feature maps (16x5x5 = 400) to 120 neurons.
  - **`fc2`**: Connects 120 neurons to 84 neurons.
  - **`fc3`**: Connects 84 neurons to 10 output classes (digits 0-9).

- **Activation Function**:
  - **`tanh`**: Applied after each convolutional and fully connected layer, introducing non-linearity.

- **Forward Pass**:
  - Data flows through convolutional layers with activation and pooling, is flattened, passes through fully connected layers with activation, and finally outputs logits for classification.

---

### 5.3. Training the Model (`train.py`)

```python
# train.py

import torch
from tqdm import tqdm

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10, log_interval=100):
    """
    Trains the given model.

    Args:
        model (nn.Module): The neural network to train.
        train_loader (DataLoader): DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run the training on.
        num_epochs (int): Number of training epochs.
        log_interval (int): How frequently to log training status.

    Returns:
        model: Trained model.
        loss_history (list): List of average losses per epoch.
    """
    model.train()  # Set model to training mode
    loss_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (inputs, labels) in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                loop.set_postfix(loss=avg_loss)
                running_loss = 0.0

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)

    print('Finished Training')
    return model, loss_history
```

**Explanation:**

- **Training Loop**:
  - **Epochs**: The number of times the entire training dataset is passed through the network.
  - **Batch Processing**: Data is processed in batches for efficient computation.
  
- **Steps Per Batch**:
  1. **Forward Pass**: Compute the model's predictions.
  2. **Loss Calculation**: Compute the difference between predictions and actual labels.
  3. **Backward Pass**: Calculate gradients of the loss with respect to model parameters.
  4. **Optimizer Step**: Update model parameters based on gradients.

- **Logging**:
  - Utilizes `tqdm` to display a progress bar and log the loss at specified intervals.

- **Loss History**:
  - Records the loss after each epoch for later visualization.

---

### 5.4. Evaluating the Model (`evaluate.py`)

```python
# evaluate.py

import torch

def evaluate_model(model, test_loader, device):
    """
    Evaluates the model's accuracy on the test dataset.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for test data.
        device: Device to run the evaluation on.

    Returns:
        accuracy (float): Overall accuracy on the test dataset.
    """
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')
    return accuracy

def evaluate_per_class(model, test_loader, device, num_classes=10):
    """
    Evaluates and prints accuracy per class.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for test data.
        device: Device to run the evaluation on.
        num_classes (int): Number of classes.
    """
    model.eval()
    class_correct = list(0. for _ in range(num_classes))
    class_total = list(0. for _ in range(num_classes))

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f'Accuracy of {i}: {accuracy:.2f}%')
        else:
            print(f'No samples for class {i}.')
```

**Explanation:**

- **`evaluate_model` Function**:
  - **Purpose**: Calculates the overall accuracy of the model on the test dataset.
  - **Process**:
    1. **Set to Evaluation Mode**: Disables dropout and batch normalization layers (if any).
    2. **Disable Gradient Calculation**: Saves memory and computation.
    3. **Prediction**: Determines the class with the highest logit score.
    4. **Accuracy Calculation**: Compares predictions with actual labels.

- **`evaluate_per_class` Function**:
  - **Purpose**: Computes and prints the accuracy for each individual class (digits 0-9).
  - **Process**:
    1. **Iterate Through Test Data**: Similar to overall evaluation.
    2. **Class-wise Counting**: Tracks correct predictions per class.
    3. **Accuracy Reporting**: Prints accuracy for each class.

---

### 5.5. Utility Functions (`utils.py`)

```python
# utils.py

import torch
import matplotlib.pyplot as plt

def save_model(model, path='lenet5.pth'):
    """
    Saves the model's state dictionary.

    Args:
        model (nn.Module): The trained model.
        path (str): File path to save the model.
    """
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def load_model(model, path='lenet5.pth', device='cpu'):
    """
    Loads the model's state dictionary.

    Args:
        model (nn.Module): The model architecture.
        path (str): File path from where to load the model.
        device: Device to map the model to.

    Returns:
        model: Model with loaded weights.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f'Model loaded from {path}')
    return model

def plot_training_loss(loss_history, num_epochs):
    """
    Plots the training loss over epochs.

    Args:
        loss_history (list): List of loss values.
        num_epochs (int): Number of epochs.
    """
    plt.figure()
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, loss_history, 'b-', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
```

**Explanation:**

- **`save_model` Function**:
  - **Purpose**: Saves the trained model's parameters to a specified path.
  - **Usage**: Useful for persisting models after training.

- **`load_model` Function**:
  - **Purpose**: Loads a saved model's parameters.
  - **Usage**: Facilitates model deployment or further evaluation without retraining.

- **`plot_training_loss` Function**:
  - **Purpose**: Visualizes the training loss over epochs.
  - **Usage**: Helps in diagnosing training performance and detecting issues like overfitting.

---

### 5.6. Main Execution Script (`main.py`)

```python
# main.py

import torch
import torch.nn as nn
import torch.optim as optim

from data import get_data_loaders, visualize_samples
from model import LeNet5
from train import train_model
from evaluate import evaluate_model, evaluate_per_class
from utils import save_model, load_model, plot_training_loss

def main():
    # Hyperparameters
    batch_size = 64
    test_batch_size = 1000
    learning_rate = 0.01
    momentum = 0.9
    num_epochs = 10
    log_interval = 100
    model_path = 'lenet5.pth'
    data_dir = './data'  # Ensure this path points to the directory containing 'train' and 'test' folders

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Data loaders
    train_loader, test_loader = get_data_loaders(data_dir=data_dir, batch_size=batch_size, test_batch_size=test_batch_size)

    # (Optional) Visualize some training samples
    # visualize_samples(train_loader, classes=[str(i) for i in range(10)])

    # Initialize the model
    model = LeNet5().to(device)
    print(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Train the model and record loss history
    model, loss_history = train_model(model, train_loader, criterion, optimizer, device, num_epochs, log_interval)

    # Save the trained model
    save_model(model, model_path)

    # Plot training loss
    plot_training_loss(loss_history, num_epochs)

    # Evaluate the model
    accuracy = evaluate_model(model, test_loader, device)
    evaluate_per_class(model, test_loader, device)

    # (Optional) Load the model and evaluate again
    # loaded_model = LeNet5()
    # loaded_model = load_model(loaded_model, model_path, device)
    # evaluate_model(loaded_model, test_loader, device)

if __name__ == '__main__':
    main()
```

**Explanation:**

- **Hyperparameters**:
  - **`batch_size`**: Number of samples processed before the model is updated.
  - **`test_batch_size`**: Number of samples processed during evaluation.
  - **`learning_rate`**: Controls how much to adjust the weights with respect to the loss gradient.
  - **`momentum`**: Accelerates SGD in the relevant direction and dampens oscillations.
  - **`num_epochs`**: Total number of times the entire training dataset passes through the network.
  - **`log_interval`**: Frequency (in batches) to log training status.

- **Device Configuration**:
  - Automatically selects GPU (`cuda`) if available; otherwise, defaults to CPU.

- **Workflow**:
  1. **Data Loading**: Retrieves training and testing data loaders.
  2. **Model Initialization**: Instantiates the LeNet-5 model and moves it to the selected device.
  3. **Loss and Optimizer Setup**: Defines the loss function and optimizer.
  4. **Training**: Trains the model while recording loss history.
  5. **Model Saving**: Saves the trained model's parameters.
  6. **Loss Visualization**: Plots the training loss over epochs.
  7. **Evaluation**: Computes overall and per-class accuracy on the test dataset.
  8. **Optional Loading**: Demonstrates how to load a saved model for further evaluation.

---

## 6. Running the Project

Follow these steps to execute the project:

1. **Clone the Repository** (if applicable):
   ```bash
   git clone https://github.com/yourusername/lenet5_project.git
   cd lenet5_project
   ```

2. **Set Up a Virtual Environment** (Optional but Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Data Directory Structure**:
   - Place the MNIST `.pt` files in the following structure:
     ```
     lenet5_project/
     ├── data/
     │   ├── train/
     │   │   └── MNIST/
     │   │       ├── processed/
     │   │       │   └── training.pt
     │   │       └── raw/
     │   └── test/
     │       └── MNIST/
     │           ├── processed/
     │           │   └── test.pt
     │           └── raw/
     ```

5. **Run the Main Script**:
   ```bash
   python main.py
   ```

   **Expected Output**:
   - Progress bars indicating training progress.
   - Training loss updates at specified intervals.
   - Final training loss plot.
   - Overall test accuracy.
   - Per-class accuracy.

---

## 7. Understanding the Components

### 7.1. Convolutional Layers

**Purpose**: Extract spatial features from the input images.

**Components**:
- **Filters (Kernels)**: Small matrices (e.g., 5x5) that slide over the input image to detect features like edges, textures, etc.
- **Stride**: Determines how the filter moves across the image. A stride of 1 means the filter moves one pixel at a time.
- **Padding**: Adds borders to the input to control the spatial size of the output.

**LeNet-5 Specifics**:
- **First Convolutional Layer (`conv1`)**:
  - **Input**: 1x32x32 (grayscale image).
  - **Filters**: 6 filters of size 5x5.
  - **Output**: 6x28x28 feature maps.

- **Second Convolutional Layer (`conv2`)**:
  - **Input**: 6x14x14.
  - **Filters**: 16 filters of size 5x5.
  - **Output**: 16x10x10 feature maps.

**Mathematical Operation**:
Each filter convolves with the input, performing element-wise multiplication and summation to produce a single value in the output feature map.

### 7.2. Activation Functions

**Purpose**: Introduce non-linearity, enabling the network to learn complex patterns.

**LeNet-5's Choice**:
- **`tanh`**: Hyperbolic tangent function, chosen for its zero-centered output.

**Mathematical Definition**:
\[
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\]

**Properties**:
- Output Range: (-1, 1).
- Differentiable, facilitating gradient-based optimization.

**Modern Alternatives**:
- **ReLU (Rectified Linear Unit)**: Often preferred in contemporary networks due to faster convergence and mitigating the vanishing gradient problem.

### 7.3. Pooling Layers

**Purpose**: Reduce the spatial dimensions of the feature maps, decreasing computational load and providing spatial invariance.

**Types**:
- **Average Pooling**: Computes the average value within a window.
- **Max Pooling**: Takes the maximum value within a window.

**LeNet-5's Choice**:
- **Average Pooling**: Reflects the original architecture's design.

**Mathematical Operation**:
For a 2x2 window, average pooling computes the mean of the four values.

**Benefits**:
- Reduces the number of parameters.
- Controls overfitting by providing a summarized representation.

### 7.4. Fully Connected Layers

**Purpose**: Perform high-level reasoning by connecting every neuron in one layer to every neuron in the next.

**LeNet-5's Configuration**:
- **`fc1`**: 400 inputs (16x5x5) to 120 neurons.
- **`fc2`**: 120 neurons to 84 neurons.
- **`fc3`**: 84 neurons to 10 output classes.

**Mathematical Operation**:
Each neuron computes a weighted sum of its inputs, applies an activation function, and passes the result to the next layer.

\[
y = \phi\left(\sum_{i=1}^{N} w_i x_i + b\right)
\]

Where:
- \( \phi \) is the activation function.
- \( w_i \) are weights.
- \( x_i \) are inputs.
- \( b \) is the bias.

### 7.5. Loss Functions and Optimizers

**Loss Function**:
- **`CrossEntropyLoss`**: Suitable for multi-class classification problems. Combines `LogSoftmax` and `NLLLoss`.

**Optimizer**:
- **`SGD with Momentum`**:
  - **Stochastic Gradient Descent (SGD)**: Updates weights based on the gradient of the loss function.
  - **Momentum**: Accelerates SGD in the relevant direction and dampens oscillations.

**Mathematical Update Rule**:
\[
v_t = \gamma v_{t-1} + \eta \nabla L
\]
\[
\theta = \theta - v_t
\]

Where:
- \( v_t \): Velocity at time \( t \).
- \( \gamma \): Momentum coefficient (typically between 0.5 and 0.9).
- \( \eta \): Learning rate.
- \( \nabla L \): Gradient of the loss with respect to parameters.
- \( \theta \): Model parameters.

---

## 8. Visualizing Results

Visualizing training progress and model predictions aids in understanding and diagnosing the model's performance.

### 8.1. Plotting Training Loss

The `plot_training_loss` function in `utils.py` generates a graph showing how the loss decreases over epochs, indicating the model's learning progress.

```python
# utils.py

def plot_training_loss(loss_history, num_epochs):
    """
    Plots the training loss over epochs.

    Args:
        loss_history (list): List of loss values.
        num_epochs (int): Number of epochs.
    """
    plt.figure()
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, loss_history, 'b-', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
```

**Usage in `main.py`**:
```python
# main.py

# After training
plot_training_loss(loss_history, num_epochs)
```

**Sample Plot**:

![Training Loss](https://i.imgur.com/1R8nXyQ.png)

*Figure 3: Training Loss over Epochs*

### 8.2. Visualizing Model Predictions

The `visualize_samples` function in `data.py` allows you to see how the model's predictions align with actual labels.

```python
# main.py

# (Optional) Visualize some training samples
visualize_samples(train_loader, classes=[str(i) for i in range(10)])
```

**Sample Output**:

```
GroundTruth: 7 2 1 0 4 1 4 9
```

**Visual Output**: Displays a grid of 8 images with their corresponding ground truth labels.

---

## 9. Conclusion

This comprehensive tutorial guided you through implementing the LeNet-5 Convolutional Neural Network using PyTorch. We covered the architectural intricacies, mathematical foundations, and practical coding aspects essential for understanding and building CNNs.

**Key Takeaways**:

- **Modular Code Structure**: Enhances readability and maintainability.
- **Understanding Layers**: Grasping the role of convolutional, pooling, and fully connected layers is crucial.
- **Activation Functions**: Introduce non-linearity, enabling the network to learn complex patterns.
- **Training Mechanics**: Forward pass, loss calculation, backward pass, and optimization are the pillars of training neural networks.
- **Evaluation**: Assessing both overall and per-class accuracy provides insights into model performance.

By following this tutorial, you should now be equipped to implement, train, and evaluate CNNs for image classification tasks. Experiment with different architectures, hyperparameters, and datasets to further enhance your understanding and skills in deep learning.

---

## 10. References

1. **LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P.** (1998). *Gradient-based learning applied to document recognition*. Proceedings of the IEEE, 86(11), 2278-2324.

2. **PyTorch Documentation**: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

3. **CS231n: Convolutional Neural Networks for Visual Recognition**: [http://cs231n.stanford.edu/](http://cs231n.stanford.edu/)

4. **Deep Learning Book by Ian Goodfellow, Yoshua Bengio, and Aaron Courville**: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

5. **MNIST Dataset**: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

6. **tanh Activation Function**: [https://en.wikipedia.org/wiki/Hyperbolic_functions#Hyperbolic_tangent](https://en.wikipedia.org/wiki/Hyperbolic_functions#Hyperbolic_tangent)

7. **SGD with Momentum**: [https://towardsdatascience.com/optimizers-in-deep-learning-6683c3a7c68e](https://towardsdatascience.com/optimizers-in-deep-learning-6683c3a7c68e)

---

**Happy Coding! 🚀**