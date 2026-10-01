import time

import gymnasium as gym
import torch
import torch.nn as nn


# Same neural network structure used during training
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.network(x)


# Create CartPole with a visible window
env = gym.make("CartPole-v1", render_mode="human")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Recreate the network and load the trained model
model = DQN(state_size, action_size)

model.load_state_dict(
    torch.load(
        "cartpole_dqn_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

print("\nTesting Trained CartPole DQN Agent")
print("----------------------------------")

test_episodes = 5

for episode in range(test_episodes):

    state, info = env.reset()
    total_reward = 0
    done = False

    while not done:

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0)

        # No random exploration during testing.
        # The agent chooses its highest-valued action.
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        state, reward, terminated, truncated, info = env.step(
            action
        )

        done = terminated or truncated
        total_reward += reward

        # Slow the simulation slightly so it is easy to see.
        time.sleep(0.02)

    print(
        f"Test Episode {episode + 1}: "
        f"Reward = {total_reward:.0f}"
    )

env.close()

print("----------------------------------")
print("Testing complete.")