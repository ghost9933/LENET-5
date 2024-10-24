import tensorflow as tf
from Lenet5 import LeNet5
from data_load import load_data
from train import train
from evaluate import evaluate

def main():
    # Load the dataset
    train_dataset, test_dataset = load_data()

    # Create the model
    model = LeNet5()

    # Train the model
    train(model, train_dataset)

    # Evaluate the model
    evaluate(model, test_dataset)

if __name__ == '__main__':
    main()
