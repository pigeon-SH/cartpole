import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentNet(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.fc1 = nn.Linear(observation_space, 120).to(device)
        self.fc2 = nn.Linear(120, 84).to(device)
        self.fc3 = nn.Linear(84, action_space).to(device)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x
