import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layer_number=5):
        super(Critic, self).__init__()

        # Define input layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Define hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layer_number):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_size))  # Batch normalization
        # Define output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        x = F.relu(self.input_layer(x))

        for i in range(0, len(self.hidden_layers), 2):
            x = self.hidden_layers[i](x)  # Linear layer
            x = self.hidden_layers[i + 1](x)  # BatchNorm layer
            x = F.relu(x)  # ReLU activation
        x = self.output_layer(x)
        return x
