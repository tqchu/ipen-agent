import logging
# import torch (for neural network), etc. if needed
# from agent.replay_buffer import ReplayBuffer
# from recommender.recommender import BaseRecommender
import random

import torch
import torch.nn as nn
import torch.optim as optim

from rl.duel_q import DuelingQNetwork
from rl.replay_buffer import ReplayBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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

        self.q_network = DuelingQNetwork(state_size, action_size)
        self.target_network = DuelingQNetwork(state_size, action_size)
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
            # masked_q_values = torch.full((self.action_size,), -float('inf'))
            # valid_indices = torch.tensor(list(range(len(actions))))
            # masked_q_values[valid_indices] = q_values[0, valid_indices]

            action_idx = torch.argmax(q_values).item()
            logging.info("Choose action from experience: %d", action_idx)
        return action_idx

    def remember(self, state, action, reward, next_state, done):
        """Store an experience tuple in replay memory for later training."""
        self.memory.add(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        s, a, r, s2, d = self.memory.sample(self.batch_size)

        device = self.q_network.weight.device
        s = torch.as_tensor(s, dtype=torch.float32, device=device)
        a = torch.as_tensor(a, dtype=torch.long, device=device).unsqueeze(1)
        r = torch.as_tensor(r, dtype=torch.float32, device=device)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=device)
        d = torch.as_tensor(d, dtype=torch.float32, device=device)

        q = self.q_network(s).gather(1, a).squeeze(1)

        # Double-DQN target
        with torch.no_grad():
            online_next_a = self.q_network(s2).argmax(1, keepdim=True)
            q2_target = self.target_network(s2).gather(1, online_next_a).squeeze(1)
            y = r + self.gamma * q2_target * (1 - d)

        criterion = torch.nn.SmoothL1Loss()  # Huber
        loss = criterion(q, y)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Soft target update
        tau = 0.005
        with torch.no_grad():
            for tgt, src in zip(self.target_network.parameters(),
                                self.q_network.parameters()):
                tgt.data.mul_(1 - tau).add_(tau * src.data)

    # def train_step(self):
    #     """Perform one training step: sample a batch from memory and update the Q-network."""
    #     if len(self.memory) < self.batch_size:
    #         return
    #
    #     states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
    #     states = torch.FloatTensor(states)
    #     actions = torch.LongTensor(actions)
    #     rewards = torch.FloatTensor(rewards)
    #     next_states = torch.FloatTensor(next_states)
    #     dones = torch.FloatTensor(dones)
    #
    #     q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    #     next_q_values = self.target_network(next_states).max(1)[0]
    #     target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
    #
    #     loss = nn.MSELoss()(q_values, target_q_values.detach())
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #
    #     total_grad = 0.0
    #     for p in self.q_network.parameters():
    #         if p.grad is not None:
    #             total_grad += p.grad.abs().sum().item()
    #     logging.info(f"[step {self.step_count}] total |∇Q| = {total_grad:.6f}")
    #
    #     self.optimizer.step()
    #
    #     self.step_count += 1
    #     if self.step_count % self.update_target_freq == 0:
    #         self.update_target_network()

    def update_target_network(self):
        """Update the target network weights to match the primary Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay the exploration rate after each episode (to reduce random exploration over time)."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
