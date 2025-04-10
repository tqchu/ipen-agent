import logging
# import torch (for neural network), etc. if needed
# from agent.replay_buffer import ReplayBuffer
# from recommender.recommender import BaseRecommender
import random

import torch
import torch.nn as nn
import torch.optim as optim

from rl.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """Neural network for approximating Q-values."""

    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """
    Deep Q-Network agent that learns attack strategies via reinforcement learning.
    https://arxiv.org/abs/1312.5602
    """

    def __init__(self, state_size, action_size, recommender=None, **kwargs):
        """
        Initialize the DQN agent.
        :param state_size: Dimension of state representation.
        :param action_size: Number of possible actions.
        :param recommender: (Optional) recommender system instance for action suggestions.
        :param kwargs: Additional hyperparameters (e.g., learning rate, gamma, epsilon).
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = kwargs.get("gamma", 0.99)
        self.epsilon = kwargs.get("epsilon_start", 1.0)  # initial exploration rate
        self.epsilon_min = kwargs.get("epsilon_min", 0.1)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.995)
        self.learning_rate = kwargs.get("lr", 0.001)
        self.batch_size = kwargs.get("batch_size", 64)
        self.update_target_freq = kwargs.get("target_update", 10)
        self.memory = ReplayBuffer(capacity=kwargs.get("memory_capacity", 10000))
        self.step_count = 0

        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.recommender = recommender

        self.update_target_network()
        import logging
        logging.info("DQNAgent initialized with state_size=%d, action_size=%d", state_size, action_size)

    def select_action(self, state, actions):
        """Select an action index for the given state using an ε-greedy policy."""
        if random.random() < self.epsilon:
            action_idx = random.randrange(len(actions))
            if self.recommender is not None:
                try:
                    suggested = self.recommender.recommend(state, action_space=list(range(self.action_size)))
                    logging.debug("Recommender suggested action: %s", str(suggested))
                except NotImplementedError:
                    pass
            logging.info("Choose random action: %d", action_idx)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            masked_q_values = torch.full((self.action_size,), -float('inf'))
            valid_indices = torch.tensor(list(range(len(actions))))
            masked_q_values[valid_indices] = q_values[0, valid_indices]

            action_idx = torch.argmax(masked_q_values).item()
            logging.info("Choose action from experience: %d", action_idx)
        return action_idx

    def remember(self, state, action, reward, next_state, done):
        """Store an experience tuple in replay memory for later training."""
        self.memory.add(state, action, reward, next_state, done)

    def train_step(self):
        """Perform one training step: sample a batch from memory and update the Q-network."""
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.update_target_network()

    def update_target_network(self):
        """Update the target network weights to match the primary Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay the exploration rate after each episode (to reduce random exploration over time)."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
