import torch
import torch.nn as nn

class DuelingQNetwork(nn.Module):
    """
    We use LayerNorm to normalize the state vector for each sample.
    This ensures the scale of input features is balanced.
    Alternatively, you could preprocess the state with a running mean and std (outside the network).
    The dueling network computes a baseline value for the state and an advantage for each action.
    We combine them to get final Q-values.
    This helps the network learn that, for example, in a state with no new ports to find, the advantage of the port scan action will be low (since it adds little value over the state’s baseline).
    We would train this network with the usual DQN loss (Huber loss) and still use a target network for stability.
    The rest of the DQN algorithm (experience replay, target updates) remains as standard, but now with Double DQN logic for computing targets.
    """
    def __init__(self, state_size, action_size):
        super(DuelingQNetwork, self).__init__()
        # Optional: learnable layer norm or fixed normalization
        self.input_norm = nn.LayerNorm(state_size)  # normalize inputs

        # Common feature extraction layers
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)

        # Dueling streams: value and advantage
        self.value_fc = nn.Linear(64, 64)
        self.value_out = nn.Linear(64, 1)  # outputs scalar V(s)
        self.advantage_fc = nn.Linear(64, 64)
        self.advantage_out = nn.Linear(64, action_size)  # outputs A(s,a) for each action

    def forward(self, state):
        # Normalize state
        x = self.input_norm(state.float())
        # Common feature layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Dueling streams
        value = torch.relu(self.value_fc(x))
        value = self.value_out(value)  # shape: [batch_size, 1]
        advantage = torch.relu(self.advantage_fc(x))
        advantage = self.advantage_out(advantage)  # shape: [batch_size, action_size]
        # Combine value and advantage into Q values
        # Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        return q_values