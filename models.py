
import torch
from torch import nn
from torch.nn import functional as F

class Actor(nn.Module):
    """
    Policy Model
    """

    def __init__(self, state_size, action_size, seed):
        """
        Initialize and build the policy network.
        Args:
            state_size (int): The Dimension of states
            action_size (int): The Dimension of Action Space
            seed (int): seed
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, state):
        """
        Forward path of the policy network.
        Args:
            state: tensor of states

        Returns:
            actions
        """
        x = state
        x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))



class Critic(nn.Module):
    """
    Critic Model
    """

    def __init__(self, state_size, action_size, seed):
        """
        Initialize and build the critic network.
        Args:
            state_size (int): The Dimension of states
            action_size (int): The Dimension of Action Space
            seed (int): seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcState = nn.Linear(state_size, 32)
        self.fcAction = nn.Linear(action_size, 16)
        self.fc2 = nn.Linear(32+16, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, state, action):
        """
        Forward path of the critic network.
        Args:
            state: tensor of states
            action: tensor of actions

        Returns:
            qValue: Q(state,action)
        """
        xState = F.relu(self.fcState(state))
        xAction = F.relu(self.fcAction(action))
        x = torch.cat((xState, xAction), 1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
