from collections import deque
import random
import numpy as np

from rl.replay_buffer import ReplayBuffer

# 1. Create ReplayBuffer with capacity 10
replay_buffer = ReplayBuffer(capacity=10)

# 2. Add some experiences
for i in range(15):
    # For demonstration, states and next_states can be small NumPy arrays
    state = np.array([i, i + 0.5])
    action = i % 3
    reward = float(i)
    next_state = state + 1
    done = (i % 5 == 0)  # Mark as done every 5 steps, just as an example
    replay_buffer.add(state, action, reward, next_state, done)
    print(
        f"Added transition #{i} -> (state={state}, action={action}, reward={reward}, next_state={next_state}, done={done})")

# 3. Check the current size of the buffer
print(f"\nBuffer size after adding 15 transitions (capacity=10): {len(replay_buffer)}")

sample_size = 5
if len(replay_buffer) >= sample_size:
    states, actions, rewards, next_states, dones = replay_buffer.sample(sample_size)
    print(f"\nSampled {sample_size} transitions:")
    for idx in range(sample_size):
        print(f"  [{idx}] state={states[idx]}, action={actions[idx]}, "
              f"reward={rewards[idx]}, next_state={next_states[idx]}, done={dones[idx]}")
else:
    print(f"Not enough data to sample {sample_size} items from the buffer.")

