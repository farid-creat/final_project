import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable
class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layer_number=4):
        super(Actor, self).__init__()

        # Define input layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Define hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layer_number):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_size))  # Batch normalization

        # Define output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        for i in range(0, len(self.hidden_layers), 2):
            x = self.hidden_layers[i](x)  # Linear layer
            if x.size(0) > 1:  # Check if batch size > 1
                x = self.hidden_layers[i + 1](x)  # BatchNorm layer
            x = F.relu(x)  # ReLU activation
        x = self.output_layer(x)
        return x

# Example usage:
# actor = Actor(input_size=10, hidden_size=64, output_size=5, hidden_layer_number=4)
# state = torch.randn(1, 10)
# action = actor(state)



