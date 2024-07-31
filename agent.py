import torch
import pygame
import numpy as np
import random
from environment import (
    SantaFeEnvironment,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    CELL_WIDTH,
    CELL_HEIGHT,
    ANT_STARTING_X,
    ANT_STARTING_Y,
    DIRECTION_RIGHT,
    DIRECTION_LEFT,
    DIRECTION_UP,
    DIRECTION_DOWN,
    MAX_TIME_STEPS,
)
from collections import deque
from model import LinearQNet, QTrainer
from plotter import plot

# Constants for the agent's memory and learning
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001


class Agent:
    def __init__(self):
        self.num_runs = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(8, 256, 3)
        self.trainer = QTrainer(self.model, LEARNING_RATE, self.gamma)

    def get_state(self, environment):
        state = []

        # Determine the direction of the ant
        direction_right = environment.direction == DIRECTION_RIGHT
        direction_left = environment.direction == DIRECTION_LEFT
        direction_up = environment.direction == DIRECTION_UP
        direction_down = environment.direction == DIRECTION_DOWN

        # Initialize food pellet positions as False
        food_pellet_right = False
        food_pellet_left = False
        food_pellet_up = False
        food_pellet_down = False

        # Calculate current ant position
        current_ant_position = (
            int(environment.x // CELL_WIDTH),
            int(environment.y // CELL_HEIGHT),
        )

        # If there are no food pellets left, populate the state vector and return it
        if len(environment.food_pellet_positions) == 0:
            state = [
                direction_right,
                direction_left,
                direction_up,
                direction_down,
                food_pellet_right,
                food_pellet_left,
                food_pellet_up,
                food_pellet_down,
            ]

            return np.array(state, dtype=int)

        # Calculate the nearest food pellet to the ant
        nearest_food_pellet = min(
            environment.food_pellet_positions,
            key=lambda food_pellet: abs(
                food_pellet[0] - current_ant_position[0]
            )
            + abs(food_pellet[1] - current_ant_position[1]),
        )
        food_pellet_x, food_pellet_y = nearest_food_pellet

        # Determine the relative position of the nearest food pellet
        food_pellet_right = food_pellet_x > current_ant_position[0]
        food_pellet_left = food_pellet_x < current_ant_position[0]
        food_pellet_up = food_pellet_y < current_ant_position[1]
        food_pellet_down = food_pellet_y > current_ant_position[1]

        # Populate the state vector and return it
        state = [
            direction_right,
            direction_left,
            direction_up,
            direction_down,
            food_pellet_right,
            food_pellet_left,
            food_pellet_up,
            food_pellet_down,
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, run):
        # Store experience tuple in memory
        self.memory.append((state, action, reward, next_state, run))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        # Seperate the mini-sample into states, actions, rewards, next_states, and runs
        states, actions, rewards, next_states, runs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, runs)

    def train_short_memory(self, state, action, reward, next_state, run):
        # Train the model with a single step of experience
        self.trainer.train_step(state, action, reward, next_state, run)

    def get_action(self, state):
        self.epsilon = 80 - self.num_runs
        final_move = [0, 0, 0]

        # Determine the action to take based on the epsilon-greedy policy (exploration vs. exploitation)
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    # Initialize variables for plotting
    plot_scores = []
    plot_mean_scores = []

    plot_time_steps = []
    plot_mean_time_steps = []

    total_score = 0
    total_time_steps = 0

    max_score = 0
    min_time_steps = MAX_TIME_STEPS

    # Initialize the agent and the environment
    agent = Agent()
    environment = SantaFeEnvironment(WINDOW_WIDTH, WINDOW_HEIGHT)

    try:
        while True:
            # Get the current state of the environment
            state_old = agent.get_state(environment)

            # Get the action to take based on the current state
            final_move = agent.get_action(state_old)

            # Perform the action and get the reward and next state
            run, reward, score, time_steps = environment.step(final_move)
            state_new = agent.get_state(environment)

            # Train the agent with short memory
            agent.train_short_memory(
                state_old, final_move, reward, state_new, run
            )

            # Store the experience in the agent's memory
            agent.remember(state_old, final_move, reward, state_new, run)

            if not run:
                # Reset the environment
                environment.reset(
                    ANT_STARTING_X, ANT_STARTING_Y, DIRECTION_RIGHT
                )
                agent.num_runs += 1

                # Train the agent with long memory
                agent.train_long_memory()

                # Update the max score and min time steps
                if score > max_score:
                    max_score = score

                if time_steps < min_time_steps:
                    min_time_steps = time_steps

                # Print the current run's statistics
                print(
                    f"Run #{agent.num_runs} - Score: {score}, Highest Score: {max_score} | "
                    f"Time Steps: {time_steps}, Lowest Time Steps: {min_time_steps}"
                )

                # Update the score and mean score data for plotting
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.num_runs
                plot_mean_scores.append(mean_score)

                # Update the time steps and mean time steps data for plotting
                plot_time_steps.append(time_steps)
                total_time_steps += time_steps
                mean_time_steps = total_time_steps / agent.num_runs
                plot_mean_time_steps.append(mean_time_steps)

                # Plot the scores and time steps
                plot(
                    plot_scores,
                    plot_mean_scores,
                    plot_time_steps,
                    plot_mean_time_steps,
                )
    except pygame.error:
        return


if __name__ == "__main__":
    train()
