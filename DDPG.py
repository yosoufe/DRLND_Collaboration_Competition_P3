import torch
from torch import nn
import torch.optim as optim
from tools import ReplayBuffer, OUNoise


class DDPG_Agent:
    """
    DDPG Algorithm
    """

    def __init__(self,
                 state_size,
                 action_size,
                 actor_model,
                 critic_model,
                 device,
                 num_agents=1,
                 seed=0,
                 tau=1e-3,
                 batch_size=1024,
                 discount_factor=0.99,
                 actor_learning_rate=1e-4,
                 critic_learning_rate=1e-3):
        """
        Initialize the 4 networks
        Copy 2 of them into the other two:
        * actor and actor_target
        * critic and critic_target
        init the replay buffer and the noise process

        Args:
            state_size:
            action_size:
            num_agents:
            seed:
            tau:
            batch_size:
            discount_factor:
            actor_learning_rate:
            critic_learning_rate:

        """
        self.tau = tau
        self.state_size = state_size
        self.action_size = action_size
        self.actor_local = actor_model(state_size, action_size, seed)
        self.actor_target = actor_model(state_size, action_size, seed)
        self.critic_local = critic_model(state_size, action_size, seed)
        self.critic_target = critic_model(state_size, action_size, seed)
        self.critic2_local = critic_model(state_size, action_size, seed+1)
        self.critic2_target = critic_model(state_size, action_size, seed+1)
        self.soft_update(1.0)
        self.batch_size = batch_size
        self.replayBuffer = ReplayBuffer(batch_size=batch_size, buffer_size=300*1000, seed=seed, device=device)
        self.num_agents = num_agents
        self.noise_process = OUNoise(action_size * num_agents, seed, max_sigma=0.1, min_sigma=0.001, decay_period=300*300)
        self.discount_factor = discount_factor
        self.actor_opt = optim.Adam(self.actor_local.parameters(), lr=actor_learning_rate)
        self.critic_opt = optim.Adam(self.critic_local.parameters(), lr=critic_learning_rate)
        self.critic2_opt = optim.Adam(self.critic2_local.parameters(), lr=critic_learning_rate)
        self.critic_criterion = nn.MSELoss()
        self.critic2_criterion = nn.MSELoss()
        self.device = device
        for model in [self.actor_local,
                      self.actor_target,
                      self.critic_local,
                      self.critic_target,
                      self.critic2_local,
                      self.critic2_target]:
            model.to(device)

    def act(self, state, add_noise=True):
        """
        * Create actions using Actor Policy Network
        * Add noise to the actions and return it.

        Args:
            state: numpy array in shape of (num_agents, action_size).
            add_noise:

        Returns:
            actions_with_noise: numpy arrays of size (num_agents, action_size)
            actions_without_noise: numpy arrays of size (num_agents, action_size)
        """
        state = torch.from_numpy(state).float().view(self.num_agents, self.state_size).to(self.device)
        self.actor_local.eval()
        actions_with_noise = None
        actions_without_noise = None
        with torch.no_grad():
            actions = self.actor_local(state)
            actions_without_noise = actions.cpu().numpy()
        self.actor_local.train()
        if add_noise:
            actions_with_noise = actions_without_noise + self.noise_process.sample().reshape(self.num_agents, self.action_size)
        return actions_with_noise, actions_without_noise

    def step(self, state, action, reward, next_state, done):
        """
        * save sample in the replay buffer
        * if replay buffer is large enough
            * learn

        Args:
            state:
            action:
            reward:
            next_state:
            done:

        Returns:
            None
        """
        self.replayBuffer.push(state, action, reward, next_state, done)
        if len(self.replayBuffer) > self.batch_size:
            self.learn(*self.replayBuffer.sample())

    def learn(self, states, actions, rewards, next_states, dones):
        """
        * sample a batch
        * set y from reward, Target Critic Network and Target Policy network
        * Calculate loss from y and Critic Network
        * Update the actor policy (would also update the critic by chain rule) using sampled policy gradient
        * soft update the target critic and target policy

        Args:
            actions:
            rewards:
            next_states:
            dones:

        Returns:
            None
        """
        # Update Critic
        next_actions = self.actor_target(next_states)
        # value = self.critic_target(next_states, next_actions).detach()
        value = (self.critic_target(next_states, next_actions).detach() +
                 self.critic2_target(next_states, next_actions).detach()) / 2.0
        # value = torch.min(self.critic_target(next_states, next_actions).detach(),
        #                   self.critic2_target(next_states, next_actions).detach())
        y = rewards + self.discount_factor * value

        Q = self.critic_local(states, actions)
        critic_loss = self.critic_criterion(Q, y)

        Q2 = self.critic2_local(states, actions)
        critic2_loss = self.critic2_criterion(Q2, y)

        # Update Actor
        action_predictions = self.actor_local(states)
        actor_loss = -self.critic_local(states, action_predictions).mean()

        # update networks
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # soft update
        self.soft_update(self.tau)

    def reset(self):
        self.noise_process.reset()

    def soft_update(self, tau):
        for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.critic2_target.parameters(), self.critic2_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class DDPG_interactive:
    """
    DDPG for multi and interactive agent Algorithm
    """

    def __init__(self,
                 state_size,
                 action_size,
                 actor_model,
                 critic_model,
                 device,
                 num_agents=2,
                 seed=0,
                 tau=1e-3,
                 batch_size=1024,
                 discount_factor=0.99,
                 actor_learning_rate=1e-4,
                 critic_learning_rate=1e-3):
        """
        Initialize the 4 networks
        Copy 2 of them into the other two:
        * actor and actor_target
        * critic and critic_target
        init the replay buffer and the noise process

        Args:
            state_size:
            action_size:
            num_agents:
            seed:
            tau:
            batch_size:
            discount_factor:
            actor_learning_rate:
            critic_learning_rate:

        """
        self.tau = tau
        self.state_size = state_size
        self.action_size = action_size
        self.actor_local = actor_model(state_size, action_size, seed)
        self.actor_target = actor_model(state_size, action_size, seed)
        self.critic_local = critic_model(state_size*num_agents, action_size*num_agents, seed)
        self.critic_target = critic_model(state_size*num_agents, action_size*num_agents, seed)
        self.critic2_local = critic_model(state_size*num_agents, action_size*num_agents, seed+1)
        self.critic2_target = critic_model(state_size*num_agents, action_size*num_agents, seed+1)
        self.soft_update(1.0)
        self.batch_size = batch_size
        self.replayBuffer = ReplayBuffer(batch_size=batch_size, buffer_size=300*1000, seed=seed, device=device)
        self.num_agents = num_agents
        self.noise_process = OUNoise(action_size * num_agents, seed, max_sigma=0.1, min_sigma=0.001, decay_period=300*300)
        self.discount_factor = discount_factor
        self.actor_opt = optim.Adam(self.actor_local.parameters(), lr=actor_learning_rate)
        self.critic_opt = optim.Adam(self.critic_local.parameters(), lr=critic_learning_rate)
        self.critic2_opt = optim.Adam(self.critic2_local.parameters(), lr=critic_learning_rate)
        self.critic_criterion = nn.MSELoss()
        self.critic2_criterion = nn.MSELoss()
        self.device = device
        for model in [self.actor_local,
                      self.actor_target,
                      self.critic_local,
                      self.critic_target,
                      self.critic2_local,
                      self.critic2_target]:
            model.to(device)

    def act(self, state, add_noise=True):
        """
        * Create actions using Actor Policy Network
        * Add noise to the actions and return it.

        Args:
            state: numpy array in shape of (num_agents, action_size).
            add_noise:

        Returns:
            actions_with_noise: numpy arrays of size (num_agents, action_size)
            actions_without_noise: numpy arrays of size (num_agents, action_size)
        """
        state = torch.from_numpy(state).float().view(self.num_agents, self.state_size).to(self.device)
        self.actor_local.eval()
        actions_with_noise = None
        actions_without_noise = None
        with torch.no_grad():
            actions = self.actor_local(state)
            actions_without_noise = actions.cpu().numpy()
        self.actor_local.train()
        if add_noise:
            actions_with_noise = actions_without_noise + self.noise_process.sample().reshape(self.num_agents, self.action_size)
        return actions_with_noise, actions_without_noise

    def step(self, state, action, reward, next_state, done):
        """
        * save sample in the replay buffer
        * if replay buffer is large enough
            * learn

        Args:
            state:
            action:
            reward:
            next_state:
            done:

        Returns:
            None
        """
        self.replayBuffer.push(state, action, reward, next_state, done)
        if len(self.replayBuffer) > self.batch_size:
            self.learn(*self.replayBuffer.sample())

    def learn(self, states, actions, rewards, next_states, dones):
        """
        * sample a batch
        * set y from reward, Target Critic Network and Target Policy network
        * Calculate loss from y and Critic Network
        * Update the actor policy (would also update the critic by chain rule) using sampled policy gradient
        * soft update the target critic and target policy

        Args:
            actions:
            rewards:
            next_states:
            dones:

        Returns:
            None
        """
        # Update Critic
        next_actions = self.actor_target(next_states)
        # value = self.critic_target(next_states, next_actions).detach()
        value = (self.critic_target(next_states, next_actions).detach() +
                 self.critic2_target(next_states, next_actions).detach()) / 2.0
        # value = torch.min(self.critic_target(next_states, next_actions).detach(),
        #                   self.critic2_target(next_states, next_actions).detach())
        y = rewards + self.discount_factor * value

        Q = self.critic_local(states, actions)
        critic_loss = self.critic_criterion(Q, y)

        Q2 = self.critic2_local(states, actions)
        critic2_loss = self.critic2_criterion(Q2, y)

        # Update Actor
        action_predictions = self.actor_local(states)
        actor_loss = -self.critic_local(states, action_predictions).mean()

        # update networks
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # soft update
        self.soft_update(self.tau)

    def reset(self):
        self.noise_process.reset()

    def soft_update(self, tau):
        for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.critic2_target.parameters(), self.critic2_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
