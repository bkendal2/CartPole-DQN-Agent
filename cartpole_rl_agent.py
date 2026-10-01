import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Deep Q-Network
# -----------------------------
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


# -----------------------------
# Replay Memory
# -----------------------------
class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# -----------------------------
# DQN Agent
# -----------------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64

        self.memory = ReplayMemory()

        self.policy_network = DQN(
            state_size,
            action_size
        )

        self.target_network = DQN(
            state_size,
            action_size
        )

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate
        )

        self.loss_function = nn.MSELoss()

    def choose_action(self, state):
        # Exploration: choose a random action
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Exploitation: choose the action with the
        # highest predicted Q-value
        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_network(
                state_tensor
            )

        return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(
            self.batch_size
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        states = torch.FloatTensor(
            np.array(states)
        )

        actions = torch.LongTensor(
            actions
        ).unsqueeze(1)

        rewards = torch.FloatTensor(
            rewards
        ).unsqueeze(1)

        next_states = torch.FloatTensor(
            np.array(next_states)
        )

        dones = torch.FloatTensor(
            dones
        ).unsqueeze(1)

        current_q_values = self.policy_network(
            states
        ).gather(1, actions)

        with torch.no_grad():
            max_next_q_values = self.target_network(
                next_states
            ).max(1)[0].unsqueeze(1)

            target_q_values = rewards + (
                self.gamma
                * max_next_q_values
                * (1 - dones)
            )

        loss = self.loss_function(
            current_q_values,
            target_q_values
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )


# -----------------------------
# Training
# -----------------------------
def train_agent():
    env = gym.make("CartPole-v1")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(
        state_size,
        action_size
    )

    episodes = 500
    scores = []

    print("\nCartPole DQN Reinforcement Learning Agent")
    print("-----------------------------------------")
    print("State features:", state_size)
    print("Possible actions:", action_size)
    print("Training episodes:", episodes)
    print("-----------------------------------------\n")

    for episode in range(episodes):
        state, info = env.reset()

        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)

            next_state, reward, terminated, truncated, info = env.step(
                action
            )

            done = terminated or truncated

            agent.memory.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            agent.train()

            state = next_state
            total_reward += reward

        scores.append(total_reward)

        # Reduce random exploration over time
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # Periodically update the target network
        if (episode + 1) % 10 == 0:
            agent.update_target_network()

        # Display training progress
        if (episode + 1) % 10 == 0:
            recent_average = np.mean(
                scores[-10:]
            )

            print(
                f"Episode {episode + 1:3d} | "
                f"Reward: {total_reward:6.1f} | "
                f"10-Episode Avg: {recent_average:6.1f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    env.close()

    # Save the trained model
    torch.save(
        agent.policy_network.state_dict(),
        "cartpole_dqn_model.pth"
    )

    # Calculate final statistics
    first_50_average = np.mean(scores[:50])
    last_50_average = np.mean(scores[-50:])
    best_score = np.max(scores)

    print("\nTraining Complete!")
    print("-----------------------------------------")
    print(
        f"Average reward - first 50 episodes: "
        f"{first_50_average:.2f}"
    )
    print(
        f"Average reward - last 50 episodes: "
        f"{last_50_average:.2f}"
    )
    print(
        f"Best episode reward: {best_score:.2f}"
    )
    print("-----------------------------------------")

    # -----------------------------
    # Create Results Graph
    # -----------------------------
    plt.figure(figsize=(10, 6))

    plt.plot(
        scores,
        label="Episode Reward",
        alpha=0.5
    )

    if len(scores) >= 20:
        moving_average = np.convolve(
            scores,
            np.ones(20) / 20,
            mode="valid"
        )

        plt.plot(
            range(19, len(scores)),
            moving_average,
            label="20-Episode Moving Average",
            linewidth=2
        )

    plt.title(
        "CartPole DQN Training Performance"
    )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "cartpole_training_results.png",
        dpi=300
    )

    print(
        "\nTraining graph saved as "
        "'cartpole_training_results.png'"
    )

    plt.show()


if __name__ == "__main__":
    train_agent()