import torch

from collections import deque, namedtuple
import random
import numpy as np
import copy

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class ReplayBuffer:
    """
    Replay Buffer

    Replay Buffer saves tuples of (s_t, a_t, r_t, s_(t+1), done) in circular buffer.
    Then it samples from them uniformly.
    """

    def __init__(self, batch_size, buffer_size=100000, seed=0, device='cpu'):
        self.buffer = deque(maxlen=buffer_size)
        self.Experience = namedtuple("Experience",
                                     field_names=["state",
                                                  "action",
                                                  "reward",
                                                  "next_state",
                                                  "done"])
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.device = device

    def push(self, state, action, reward, next_states, done):
        """
        Add an experience to the buffer.

        Args:
            state: 2D numpy array in size of number of agents by number of states
            action: 2D numpy array in size of number of agents by number of actions
            reward: list in size of number of agents
            next_states: same as state
            done (boolean): list in size of number of agents

        Returns:
            None
        """
        for i in range(state.shape[0]):
            ex = self.Experience(state[i, :], action[i, :], reward[i], next_states[i, :], done[i])
            self.buffer.append(ex)

    def sample(self):
        """
        sample from the buffer

        Returns:
            state: tensor in size (batch_size, number of states)
            action: tensor in size (batch_size, number of actions)
            reward: tensor in size (batch_size, 1)
            next_state: same as state
            done (boolean): tensor in size (batch_size, number of states)
        """
        ex = random.sample(self.buffer, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in ex if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in ex if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in ex if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in ex if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in ex if e is not None]).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """

        Returns:
            number of experiences in the buffer
        """
        return len(self.buffer)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, max_sigma=0.1, min_sigma=0.01, decay_period=100000):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.seed = random.seed(seed)
        self.step = 0
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.step += 1
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, self.step / self.decay_period)
        self.state = x + dx
        return self.state


class PlotTool:
    """
    Example:
        from tools import PlotTool
        from time import sleep
        %matplotlib notebook
        plotter = PlotTool(number_of_lines=5)

        for i in range(3000):
            plotter.update_plot(np.random.randn(1,5))
            sleep(0.001)
    """

    def __init__(self, number_of_lines, desc, number_of_actions=0):
        self.number_of_lines = number_of_lines
        self.rewards = np.empty((number_of_lines, 1), dtype=np.float)
        self.average = np.empty((1, 1), dtype=np.float)
        self.initialized = False
        self.fig = plt.figure(figsize=(8, 16))
        self.axRews = self.fig.add_subplot(2, 1, 1)
        self.axAve = self.fig.add_subplot(2, 1, 2)
        self.axRews.autoscale()
        self.axAve.autoscale()
        self.actions = None
        self.desc = desc
        if number_of_actions:
            self.axRews = self.fig.add_subplot(3, 1, 1)
            self.axAve = self.fig.add_subplot(3, 1, 2)
            self.actions = self.fig.add_subplot(3, 1, 3)
            self.axRews.autoscale()
            self.axAve.autoscale()
            self.actions.autoscale()

        self.lines = [Line2D([0], [0], linestyle='-') for _ in range(number_of_lines)]
        for i, line in enumerate(self.lines):
            self.axRews.add_line(line)
            line.set_color((random.random(), random.random(), random.random()))
            line.set_label('A{}{}'.format(self.desc,i))
        if number_of_actions:
            self.lines = [Line2D([0], [0], linestyle='-') for _ in range(number_of_actions)]
            for i, line in enumerate(self.lines):
                self.actions.add_line(line)
                line.set_color((random.random(), random.random(), random.random()))
                line.set_label('A{}{}'.format(self.desc,i))
        assert len(self.lines) == number_of_lines

        self.average_line = Line2D([0], [0], linestyle='-')
        self.average_line.set_label('Ave{}'.format(self.desc))
        self.average_line.set_color((0, 0, 0))
        self.axAve.add_line(self.average_line)

        self.axRews.legend(bbox_to_anchor=(.99, 1), loc='upper left', ncol=1)
        self.axAve.legend(bbox_to_anchor=(.99, 1), loc='upper left', ncol=1)

    def push_date(self, newRewards):
        newRewards = newRewards.reshape(1, self.number_of_lines)
        if not self.initialized:
            self.rewards = newRewards
            self.average = np.mean(self.rewards, axis=1).reshape((1, 1))
            self.initialized = True
        else:
            self.rewards = np.concatenate((self.rewards, newRewards), axis=0)
            self.average = np.concatenate((self.average, np.mean(newRewards).reshape((1, 1))), axis=0)

            for li, line in enumerate(self.lines):
                line.set_xdata(np.arange(self.rewards.shape[0]))
                line.set_ydata(self.rewards[:, li])
            self.average_line.set_xdata(np.arange(self.average.shape[0]))
            self.average_line.set_ydata(self.average)

    def draw(self, reset_for_next_time=False):
        self.axRews.relim()
        self.axRews.autoscale_view()
        self.axAve.relim()
        self.axAve.autoscale_view()
        self.fig.canvas.draw()
        if reset_for_next_time:
            self.initialized = False

