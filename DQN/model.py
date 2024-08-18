"""
Script that contains details about the neural network model used for the DQN Agent
"""
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQNNet(nn.Module):
    """
    Class that defines the architecture of the neural network for the DQN agent
    """
    def __init__(self, input_size, output_size, lr=1e-3):
        super(DQNNet, self).__init__()
        self.dense1 = nn.Linear(input_size, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 256)
        self.dense5 = nn.Linear(256, 256)
        self.dense6 = nn.Linear(256, 128)
        self.dense7 = nn.Linear(128, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
        x = F.relu(self.dense5(x))
        x = F.relu(self.dense6(x))
        x = self.dense7(x)
        return x

    def save_model(self, path , name):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, f'{name}.pth'))

    def load_model(self, path, device , name):
        self.load_state_dict(torch.load(os.path.join(path, f'{name}.pth'), map_location=device))



