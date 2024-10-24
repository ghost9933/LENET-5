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
