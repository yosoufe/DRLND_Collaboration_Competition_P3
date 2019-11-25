import torch
from torch import nn
from torch.nn import functional as F


class MACritic(nn.Module):
    """
    Critic Model
    """

    def __init__(self,
                 state_size,
                 other_states,
                 action_size,
                 other_actions,
                 seed,
                 layer_sizes=[64, 16, 256, 32]):
        """
        Initialize and build the critic network.
        Args:
            state_size (int): The Dimension of states
            action_size (int): The Dimension of Action Space
            seed (int): seed
        """
        self.state_size=state_size
        self.other_states=other_states
        self.action_size=action_size
        self.other_actions=other_actions

        super(MACritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn_s = nn.BatchNorm1d(state_size)
        self.bn_so = nn.BatchNorm1d(other_states)
        self.fcState = nn.Linear(state_size, layer_sizes[0])
        self.fcAction = nn.Linear(action_size, layer_sizes[1])

        self.fcState_o = nn.Linear(other_states, layer_sizes[0])
        self.fcAction_o = nn.Linear(other_actions, layer_sizes[1])

        self.fc2 = nn.Linear(layer_sizes[0] + layer_sizes[1], layer_sizes[2])
        self.fc2_o = nn.Linear(layer_sizes[0] + layer_sizes[1], layer_sizes[2])
        self.fc3 = nn.Linear(layer_sizes[2]*2, layer_sizes[3])
        self.fc4 = nn.Linear(layer_sizes[3], 1)

    def forward(self, state, action):
        """
        Forward path of the critic network.
        Args:
            state: tensor of states
            action: tensor of actions

        Returns:
            qValue: Q(state,action)
        """
        this_state = state[:, :self.state_size]
        other_state = state[:, self.state_size:]
        this_action = action[:, :self.action_size]
        other_action = action[:, self.action_size:]


        this_state = self.bn_s(this_state)
        other_state = self.bn_so(other_state)

        this_state = F.relu(self.fcState(this_state))
        this_action = F.relu(self.fcAction(this_action))
        other_state = F.relu(self.fcState_o(other_state))
        other_action = F.relu(self.fcAction_o(other_action))

        this_v = F.relu(self.fc2(torch.cat((this_state, this_action), 1)))
        other_v = F.relu(self.fc2_o(torch.cat((other_state, other_action), 1)))

        v = F.relu(self.fc3(torch.cat((this_v, other_v), 1)))
        return self.fc4(v)


class Actor(nn.Module):
    """
    Policy Model
    """

    def __init__(self, state_size, action_size, seed, layer_sizes=[32, 64, 64]):
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
        self.fc1 = nn.Linear(state_size, layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.fc4 = nn.Linear(layer_sizes[2], action_size)

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

    def __init__(self, state_size, action_size, seed, layer_sizes=[32, 16, 128, 32]):
        """
        Initialize and build the critic network.
        Args:
            state_size (int): The Dimension of states
            action_size (int): The Dimension of Action Space
            seed (int): seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcState = nn.Linear(state_size, layer_sizes[0])
        self.fcAction = nn.Linear(action_size, layer_sizes[1])
        self.fc2 = nn.Linear(layer_sizes[0] + layer_sizes[1], layer_sizes[2])
        self.fc3 = nn.Linear(layer_sizes[2], layer_sizes[3])
        self.fc4 = nn.Linear(layer_sizes[3], 1)

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
