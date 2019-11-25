import torch
from torch import nn
import torch.optim as optim
from tools import ReplayBuffer, OUNoise
import os.path
import numpy as np


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
        self.critic2_local = critic_model(state_size, action_size, seed + 1)
        self.critic2_target = critic_model(state_size, action_size, seed + 1)
        self.soft_update(1.0)
        self.batch_size = batch_size
        self.replayBuffer = ReplayBuffer(batch_size=batch_size, buffer_size=300 * 1000, seed=seed, device=device)
        self.num_agents = num_agents
        self.noise_process = OUNoise(action_size * num_agents, seed, max_sigma=0.3, min_sigma=0.001,
                                     decay_period=300*300)
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
            actions_with_noise = actions_without_noise + self.noise_process.sample().reshape(self.num_agents,
                                                                                             self.action_size)
        return actions_with_noise, actions_without_noise

    def step(self, state, action, reward, next_state, done):
        """
        * save sample in the replay buffer
        * if replay buffer is large enough
            * learn

        Args:
            state: in size of number_agents by number of states
            action: in size of number_agents by number of actions
            reward: list in size of num_agents
            next_state: same as state
            done: list of booleans in size of num_agents

        Returns:
            None
        """

        for i in range(state.shape[0]):
            self.replayBuffer.push(state[i, :].reshape(1, -1),
                                   action[i, :].reshape(1, -1),
                                   reward[i],
                                   next_state[i, :].reshape(1, -1),
                                   None,
                                   done[i])

        if len(self.replayBuffer) > self.batch_size:
            self.learn(*self.replayBuffer.sample())

    def learn(self, states, actions, rewards, next_states, next_actions, dones):
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
            next_actions:
            dones:

        Returns:
            None
        """
        # Update Critic
        if next_actions is None:
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

    def save_agent(self, file_name):
        i = 0
        path = None
        while True:
            if not os.path.isfile(f'{file_name}_{i}.pth'):
                path = f'{file_name}_{i}.pth'
                break
            else:
                i += 1

        torch.save({
            'actor_local': self.actor_local.state_dict(),
            'critic_local': self.critic_local.state_dict(),
            'critic2_local': self.critic2_local.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'critic2_opt': self.critic2_opt.state_dict()
        }, path)

    def load_agent(self, file_name, is_exact_path=False):
        i = 0
        path = None
        while True:
            if os.path.isfile(f'{file_name}_{i}.pth'):
                path = f'{file_name}_{i}.pth'
                i += 1
            else:
                break

        ckpt = torch.load(path)
        self.actor_local.load_state_dict(ckpt['actor_local'])
        self.critic_local.load_state_dict(ckpt['critic_local'])
        self.critic2_local.load_state_dict(ckpt['critic2_local'])
        self.actor_opt.load_state_dict(ckpt['actor_opt'])
        self.critic_opt.load_state_dict(ckpt['critic_opt'])
        self.critic2_opt.load_state_dict(ckpt['critic2_opt'])
        self.actor_local.to(self.device)
        self.critic_local.to(self.device)
        self.critic2_local.to(self.device)


class MADDPG:
    """
    DDPG for multi and interactive agent Algorithm
    """

    def __init__(self,
                 state_size,
                 action_size,
                 actor_model,
                 critic_model,
                 device,
                 num_agents=1,
                 num_interacting_agents=2,
                 seed=0,
                 tau=1e-3,
                 batch_size=1024,
                 discount_factor=0.99,
                 actor_learning_rate=1e-4,
                 critic_learning_rate=1e-3,
                 replayBuffer=None):
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
        actor_layers = [32, 64, 64]
        critic_layers = [64, 16, 128, 16]  # [64, 64, 256, 32]
        self.actor_local = actor_model(state_size, action_size, seed, layer_sizes=actor_layers)
        self.actor_target = actor_model(state_size, action_size, seed, layer_sizes=actor_layers)
        self.num_interacting_agents = num_interacting_agents
        self.critic_local = critic_model(state_size, state_size,
                                         action_size, action_size, seed,
                                         layer_sizes=critic_layers)
        self.critic_target = critic_model(state_size, state_size,
                                          action_size, action_size, seed,
                                          layer_sizes=critic_layers)
        self.critic2_local = critic_model(state_size, state_size,
                                          action_size, action_size, seed + 1,
                                          layer_sizes=critic_layers)
        self.critic2_target = critic_model(state_size, state_size,
                                           action_size, action_size, seed + 1,
                                           layer_sizes=critic_layers)
        self.soft_update(1.0)
        self.batch_size = batch_size
        if replayBuffer is None:
            self.replayBuffer = ReplayBuffer(batch_size=batch_size, buffer_size=300 * 1000, seed=seed, device=device)
        else:
            self.replayBuffer = replayBuffer
        self.num_agents = num_agents
        self.noise_process = OUNoise(action_size * num_agents, seed, max_sigma=0.1, min_sigma=0.001,
                                     decay_period=30*300,
                                     decay_delay=4048/30)
        self.discount_factor = discount_factor
        self.actor_opt = optim.Adam(self.actor_local.parameters(), lr=actor_learning_rate)
        self.critic_opt = optim.Adam(self.critic_local.parameters(), lr=critic_learning_rate)
        self.critic2_opt = optim.Adam(self.critic2_local.parameters(), lr=critic_learning_rate)
        self.critic_criterion = nn.MSELoss()
        self.critic2_criterion = nn.MSELoss()
        self.device = device
        self.other = None
        for model in [self.actor_local,
                      self.actor_target,
                      self.critic_local,
                      self.critic_target,
                      self.critic2_local,
                      self.critic2_target]:
            model.to(device)

    def set_other_agent(self, agent):
        self.other = agent

    def act(self, state, add_noise=True):
        """
        * Create actions using Actor Policy Network
        * Add noise to the actions and return it.

        Args:
            state: numpy array in shape of (state_size,).
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
            actions_with_noise = actions_without_noise + self.noise_process.sample().reshape(self.num_agents,
                                                                                             self.action_size)
        return actions_with_noise, actions_without_noise

    def step(self,
             this_state,
             others_state,
             this_action,
             others_action,
             reward,
             this_next_states,
             others_next_states,
             done):
        """
        * save sample in the replay buffer
        * if replay buffer is large enough
            * learn

        Args:
            this_state: 1D numpy array in shape of (1, num_states)
            others_state: 1D numpy array in shape of (1, num_states*num_other_agents)
            this_action: 1D numpy array in shape of (1, num_actions)
            others_action: D numpy array in shape of (1, num_actions*num_other_agents)
            reward: reward of this agent
            this_next_states: same as this_state but for next time stamp
            others_next_states: same as others_state but for next time stamp
            this_next_actions: same as this_action but for next time stamp
            others_next_actions: same as others_action but for next time stamp
            done: of this agent

        Returns:
            None
        """
        # print(np.hstack((this_state, others_state)))
        # print(np.hstack((this_action, others_action)))
        # print(np.hstack((this_next_states, others_next_states)))
        # print(np.hstack((this_next_actions, others_next_actions)))
        # print(reward)
        self.replayBuffer.push(state=np.hstack((this_state, others_state)),
                               action=np.hstack((this_action, others_action)),
                               reward=reward,
                               next_states=np.hstack((this_next_states, others_next_states)),
                               next_actions=None,
                               done=done
                               )

        if len(self.replayBuffer) > self.batch_size * 2:
            self.learn(*self.replayBuffer.sample())

    def learn(self, states, actions, rewards, next_states, next_actions, dones):
        """
        * set y from reward, Target Critic Network and Target Policy network
        * Calculate loss from y and Critic Network
        * Update the actor policy (would also update the critic by chain rule) using sampled policy gradient
        * soft update the target critic and target policy

        Args:
            states: state of all agents in a row
            actions: actions of all agents in a row
            rewards: rewards
            next_states: next state of all agents in a row
            next_actions: same as actions
            dones: dones

        Returns:
            None
        """
        all_states = states
        this_state = states[:, 0:self.state_size]

        next_actions = self.actor_local(next_states[:, 0:self.state_size])
        next_actions = torch.cat((next_actions,
                                  self.other.actor_local(next_states[:, self.state_size:])
                                  ), 1).detach()

        # Update Critic
        value = (self.critic_target(next_states, next_actions).detach() +
                 self.critic2_target(next_states, next_actions).detach()) / 2.0
        y = rewards + self.discount_factor * value

        Q = self.critic_local(all_states, actions)
        critic_loss = self.critic_criterion(Q, y)

        Q2 = self.critic2_local(all_states, actions)
        critic2_loss = self.critic2_criterion(Q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        # Update Actor
        action_predictions = self.actor_local(this_state)
        actions_pred_and_others = torch.cat((action_predictions,
                                             next_actions[:, self.action_size:]),
                                            dim=1)
        actor_loss = (-self.critic_local(all_states, actions_pred_and_others).mean() -
                      self.critic2_local(all_states, actions_pred_and_others).mean()) * 0.5

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # soft update
        self.soft_update(self.tau)
        # print(states.shape, this_state.shape, action_predictions.shape)

    def reset(self):
        self.noise_process.reset()

    def soft_update(self, tau):
        for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.critic2_target.parameters(), self.critic2_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_agent(self, file_name):
        i = 0
        path = None
        while True:
            if not os.path.isfile(f'{file_name}_{i}.pth'):
                path = f'{file_name}_{i}.pth'
                break
            else:
                i += 1

        torch.save({
            'actor_local': self.actor_local.state_dict(),
            'critic_local': self.critic_local.state_dict(),
            'critic2_local': self.critic2_local.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'critic2_opt': self.critic2_opt.state_dict()
        }, path)

    def load_agent(self, file_name, is_exact_path=False):
        i = 0
        path = None
        while True:
            if os.path.isfile(f'{file_name}_{i}.pth'):
                path = f'{file_name}_{i}.pth'
                i += 1
            else:
                break

        ckpt = torch.load(path)
        self.actor_local.load_state_dict(ckpt['actor_local'])
        self.critic_local.load_state_dict(ckpt['critic_local'])
        self.critic2_local.load_state_dict(ckpt['critic2_local'])
        self.actor_opt.load_state_dict(ckpt['actor_opt'])
        self.critic_opt.load_state_dict(ckpt['critic_opt'])
        self.critic2_opt.load_state_dict(ckpt['critic2_opt'])
        self.actor_local.to(self.device)
        self.critic_local.to(self.device)
        self.critic2_local.to(self.device)
