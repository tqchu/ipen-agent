import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from rl.dqn import DQNAgent


# Function to train the agent
def train_dqn_agent(env, agent, num_episodes=200):
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.array(state)
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_state)

            agent.remember(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)
        agent.decay_epsilon()

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.2f}")

    return rewards


# Function to test the agent
def test_dqn_agent(env, agent, num_episodes=10, render=False):
    test_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.array(state)
        episode_reward = 0
        done = False

        while not done:
            # Always choose best action during testing (no exploration)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = agent.q_network(state_tensor)
                action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_state)

            state = next_state
            episode_reward += reward

            if render:
                env.render()

        test_rewards.append(episode_reward)
        print(f"Test Episode {episode}, Reward: {episode_reward}")

    return test_rewards


# Main function
def main():
    # Create CartPole environment
    env = gym.make('CartPole-v1')

    # Define state and action sizes
    state_size = env.observation_space.shape[0]  # 4 for CartPole
    action_size = env.action_space.n  # 2 for CartPole (left/right)

    # Initialize DQN agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        gamma=0.99,
        lr=0.001,
        memory_capacity=10000,
        batch_size=64,
        target_update=10
    )

    # Train the agent
    print("Starting training...")
    training_rewards = train_dqn_agent(env, agent, num_episodes=200)

    # Test the trained agent
    print("\nStarting testing...")
    testing_rewards = test_dqn_agent(env, agent, num_episodes=10)

    # Plot training rewards
    plt.figure(figsize=(10, 5))
    plt.plot(training_rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig('dqn_cartpole_training_rewards.png')

    print(f"Average test reward: {sum(testing_rewards) / len(testing_rewards)}")
    print("Training and testing completed!")


if __name__ == "__main__":
    main()