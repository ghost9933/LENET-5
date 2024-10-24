# main.py

import torch
import torch.nn as nn
import torch.optim as optim

from data import get_data_loaders, visualize_samples
from LeNet5 import LeNet5
from train import train_model
from eval import evaluate_model, evaluate_per_class
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
