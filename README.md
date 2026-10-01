# CartPole DQN Agent

A reinforcement learning project built in Python using a Deep Q-Network (DQN) to train an autonomous agent to balance a pole in the CartPole environment.

## Project Overview

This project demonstrates how reinforcement learning can be used to teach an AI agent through trial and error.

The agent interacts with the CartPole environment and learns which actions help keep the pole balanced. Instead of being explicitly programmed with the correct action for every situation, the agent improves its behavior over time based on rewards received from the environment.

A Deep Q-Network (DQN) is used to approximate the value of possible actions and allow the agent to learn an effective control strategy.

## Technologies Used

- Python
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- Reinforcement Learning
- Deep Q-Networks (DQN)
- Neural Networks

## How It Works

1. The CartPole environment provides the agent with the current state of the system.
2. The agent chooses an action: move the cart left or right.
3. The environment returns a reward and the next state.
4. Experiences are stored and used to train the neural network.
5. The DQN learns to estimate which actions are most valuable in different states.
6. Exploration is gradually reduced as the agent learns a stronger policy.
7. The trained model is saved and can later be loaded for testing.

## Project Files

- `cartpole_rl_agent.py` - Main reinforcement learning and DQN training program
- `test_cartpole_agent.py` - Loads and tests the trained agent
- `cartpole_dqn_model.pth` - Saved trained neural network model
- `cartpole_training_results.png` - Visualization of training performance

## Training Results

The training process tracks the agent's performance across episodes, showing how its ability to keep the CartPole balanced changes as it learns.

## Testing the Agent

After training, the saved DQN model can be loaded using the testing script. This allows the trained agent to interact with the CartPole environment using the policy it learned during training.

Run the trained agent with:

    python3 test_cartpole_agent.py

## Training the Model

To train a new agent, run:

    python3 cartpole_rl_agent.py

The program trains the DQN through repeated interactions with the environment and saves the resulting model for later testing.

## What I Learned

This project gave me hands-on experience with reinforcement learning and neural networks. I learned how an AI agent can improve its behavior through rewards, exploration, and repeated interaction with an environment.

I also gained experience implementing and training a Deep Q-Network, saving and loading trained PyTorch models, evaluating an agent after training, and visualizing training performance.

## Disclaimer

This project was created for educational and portfolio purposes.
