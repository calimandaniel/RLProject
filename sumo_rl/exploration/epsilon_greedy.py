"""Epsilon Greedy Exploration Strategy."""
import numpy as np
import torch
import matplotlib.pyplot as plt

class EpsilonGreedy:
    """Epsilon Greedy Exploration Strategy."""

    def __init__(self, initial_epsilon=1.0, min_epsilon=0.01, decay=0.99):
        """Initialize Epsilon Greedy Exploration Strategy."""
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.x = [0]
        self.y = [initial_epsilon]


    def choose(self, q_table, state, action_space):
        """Choose action based on epsilon greedy strategy."""
        if np.random.rand() < self.epsilon:
            #print("Explore")

            action = int(action_space.sample())
        else:
            #print("Exploit")

            #action = np.argmax(q_table[state])
            action = torch.argmax(q_table).item()

        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)

        return action

    def reset(self):
        """Reset epsilon to initial value."""
        self.epsilon = self.initial_epsilon



