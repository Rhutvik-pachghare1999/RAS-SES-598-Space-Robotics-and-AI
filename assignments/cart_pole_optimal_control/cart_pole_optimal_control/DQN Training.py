import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import torch.nn.functional as F
from scipy import linalg
import csv
import os

# Define the DQN Model
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

# DQN Agent Class
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.0005, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64, replay_buffer_size=100000, target_update_frequency=100, alpha=0.6, beta=0.4, max_priority=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.target_update_frequency = target_update_frequency
        self.alpha = alpha
        self.beta = beta
        self.max_priority = max_priority

        self.memory = deque(maxlen=self.replay_buffer_size)
        self.priorities = deque(maxlen=self.replay_buffer_size)

        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize the target network

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(self.max_priority)  # Initialize priorities with maximum value

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_dim))
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            actions = self.model(state_tensor)
        return torch.argmax(actions).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # Prioritized sampling
        probabilities = np.array(self.priorities) ** self.alpha
        probabilities /= probabilities.sum()  # Normalize the probabilities
        indices = random.choices(range(len(self.memory)), probabilities, k=self.batch_size)

        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in indices])

        states = torch.FloatTensor(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Calculate Q values (Double DQN)
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values using the target network and Double DQN
        with torch.no_grad():
            next_q_values = self.model(next_states).gather(1, self.model(next_states).max(1)[1].unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.criterion(q_values, target_q_values)

        # Update priorities using TD-error (|target - predicted|), for prioritized experience replay
        td_error = torch.abs(target_q_values - q_values).detach().numpy()
        for i, idx in enumerate(indices):
            self.priorities[idx] = td_error[i] + 1e-5  # Add small constant to prevent zero priorities

        # Gradient clipping to avoid exploding gradients
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Clip gradients
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        # Polyak averaging for target network update
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(0.99 * target_param.data + 0.01 * param.data)

# Prepare LQR Data for training DQN
lqr_data = pd.read_csv("/home/rhutvik/ros2_ws/src/cart_pole_optimal_control/cart_pole_optimal_control/lqr_data.csv").values
states = lqr_data[:, :4]
actions = np.clip(np.digitize(lqr_data[:, 4], bins=[-2, -1, 0, 1, 2]) - 1, 0, 4)
rewards = -np.abs(lqr_data[:, 2]) - 0.1 * np.abs(lqr_data[:, 3])
next_states = np.roll(states, -1, axis=0)
dones = np.zeros(len(states))
dones[-1] = 1

# Initialize DQN Agent and Train
agent = DQNAgent(state_dim=4, action_dim=5)
epsilons, rewards_history, theta_history = [], [], []

for i in range(len(states) - 1):
    agent.remember(states[i], int(actions[i]), rewards[i], next_states[i], bool(dones[i]))
    agent.train()

    epsilons.append(agent.epsilon)
    rewards_history.append(rewards[i])
    theta_history.append(states[i][2])

    if i % 100 == 0:
        print(f"Step {i}/{len(states) - 1}, Epsilon: {agent.epsilon:.4f}")

    if i % agent.target_update_frequency == 0:
        agent.update_target_network()

# Save the trained model
model_path = "/home/rhutvik/ros2_ws/src/cart_pole_optimal_control/cart_pole_optimal_control/dqn_cart_pole.pth"
torch.save(agent.model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Save results
results_df = pd.DataFrame(states, columns=["x", "x_dot", "theta", "theta_dot"])
results_df["action"] = actions
results_df["reward"] = rewards
results_df.to_csv("/home/rhutvik/ros2_ws/src/cart_pole_optimal_control/cart_pole_optimal_control/dqn_results.csv", index=False)

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(epsilons)
plt.title("Epsilon Decay Over Time")

plt.subplot(3, 1, 2)
plt.plot(rewards_history)
plt.title("Rewards Over Time")

plt.subplot(3, 1, 3)
plt.plot(theta_history)
plt.title("Pole Angle (Theta) Over Time")

plt.tight_layout()
plt.savefig("/home/rhutvik/ros2_ws/src/cart_pole_optimal_control/cart_pole_optimal_control/training_results.png")
plt.show()

print("Training complete, model and graphs saved.")
