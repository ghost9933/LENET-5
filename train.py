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
