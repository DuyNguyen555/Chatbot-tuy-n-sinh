import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)   # input layer to hidden layer
        self.l2 = nn.Linear(hidden_size, hidden_size)  # hidden layer to hidden layer
        self.l3 = nn.Linear(hidden_size, num_classes)  # hidden layer to output layer
        self.relu = nn.ReLU()  # activation function

    def forward(self, x):
        out = self.l1(x)      # input to hidden layer
        out = self.relu(out)  # activation function
        out = self.l2(out)    # hidden layer to hidden layer
        out = self.relu(out)  # activation function
        out = self.l3(out)    # hidden layer to output layer
        return out