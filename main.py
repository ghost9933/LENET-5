# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report

from data import get_data_loaders, visualize_samples
from LeNet5 import LeNet5
from train import train_model
from eval import evaluate_model, evaluate_per_class
from helper import save_model, load_model, plot_training_loss

def main():

    batch_size = 64
    test_batch_size = 1000
    learning_rate = 0.01
    momentum = 0.9
    num_epochs = 15
    log_interval = 100
    model_path = 'lenet5.pth'
    data_dir = './data' 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Data loaders
    train_loader, test_loader = get_data_loaders(data_dir=data_dir, batch_size=batch_size, test_batch_size=test_batch_size)

    model = LeNet5().to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


    model, loss_history = train_model(model, train_loader, criterion, optimizer, device, num_epochs, log_interval)
    save_model(model, model_path)

    plot_training_loss(loss_history, num_epochs)

    accuracy = evaluate_model(model, test_loader, device)
    evaluate_per_class(model, test_loader, device)
  
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)])

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)


if __name__ == '__main__':
    main()
