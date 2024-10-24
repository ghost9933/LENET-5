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
