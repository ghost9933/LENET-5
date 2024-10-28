import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
    
        self.convLayer1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) 
        self.poolLayer1 = nn.AvgPool2d(kernel_size=2, stride=2)  
        self.convLayer2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)  
        self.poolLayer2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fcNLayer1 = nn.Linear(16 * 5 * 5, 120)
        self.fcNlayer2 = nn.Linear(120, 84)
        self.fcNlayer3 = nn.Linear(84, 10)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        res = self.convLayer1(x)
        res = self.tanh(res)
        res = self.poolLayer1(res)
        res = self.convLayer2(res)
        res = self.tanh(res)
        res = self.poolLayer2(res)
        flatten = res.view(-1, 16 * 5 * 5)  # Flatten
        res = self.fcNLayer1(flatten)
        res = self.tanh(res)
        res = self.fcNlayer2(res)
        res = self.tanh(res)
        res = self.fcNlayer3(res)
        res = self.softmax(res)
        return res
