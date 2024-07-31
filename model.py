import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Define the layers of the neural network
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Define the forward pass
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class QTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, run):
        # Convert the experience tuples to PyTorch tensors
        state = torch.tensor(np.array(state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)

        # If a single experience tuple is passed, convert it to a batch size of 1
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            run = (run,)

        # Predict the Q-values for the current state
        prediction = self.model(state)

        # Initialize the target with the predicted Q-values
        target = prediction.clone()

        for index in range(len(run)):
            # Calculate the new Q-value based on the reward and the next state
            Q_new = reward[index]

            if run[index]:
                Q_new = reward[index] + self.gamma * torch.max(
                    self.model(next_state[index])
                )

            target[index][torch.argmax(action).item()] = Q_new

        # Zero gradients, calculate loss, perform backpropagation, and update the weights
        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        self.optimizer.step()
