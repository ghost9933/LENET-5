import torch
import matplotlib.pyplot as plt

def save_model(model, path='lenet5.pth'):
    torch.save(model.state_dict(), path)

def load_model(model, path='lenet5.pth', device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

def plot_training_loss(loss_history, num_epochs):
    plt.figure()
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, loss_history, 'b-', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
