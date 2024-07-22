import torch
import numpy as np
import random
from environment import (
    SantaFeEnvironment,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    ANT_STARTING_X,
    ANT_STARTING_Y,
    DIRECTION_RIGHT,
    DIRECTION_LEFT,
    DIRECTION_UP,
    DIRECTION_DOWN,
)
from collections import deque
from model import LinearQNet, QTrainer

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
        direction_right = environment.direction == DIRECTION_RIGHT
        direction_left = environment.direction == DIRECTION_LEFT
        direction_up = environment.direction == DIRECTION_UP
        direction_down = environment.direction == DIRECTION_DOWN

        nearest_food_pellet = min(
            environment.food_pellet_positions,
            key=lambda food_pellet: abs(food_pellet[0] - environment.x)
            + abs(food_pellet[1] - environment.y),
        )
        food_pellet_x, food_pellet_y = nearest_food_pellet

        food_pellet_right = food_pellet_x > environment.x  # food pellet right
        food_pellet_left = food_pellet_x < environment.x  # food pellet left
        food_pellet_up = food_pellet_y < environment.y  # food pellet up
        food_pellet_down = food_pellet_y > environment.y  # food pellet down

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
        self.memory.append((state, action, reward, next_state, run))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, runs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, runs)

    def train_short_memory(self, state, action, reward, next_state, run):
        self.trainer.train_step(state, action, reward, next_state, run)

    def get_action(self, state):
        self.epsilon = 80 - self.num_runs
        final_move = [0, 0, 0]

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
    plot_scores = []
    plot_mean_scores = []

    plot_time_steps = []
    plot_mean_time_steps = []

    total_score = 0
    total_time_steps = 0

    max_score = 0
    max_time_steps = 0

    agent = Agent()
    environment = SantaFeEnvironment(WINDOW_WIDTH, WINDOW_HEIGHT)

    while True:
        state_old = agent.get_state(environment)

        final_move = agent.get_action(state_old)

        run, reward, score, time_steps = environment.step(final_move)
        state_new = agent.get_state(environment)

        agent.train_short_memory(state_old, final_move, reward, state_new, run)

        agent.remember(state_old, final_move, reward, state_new, run)

        if not run:
            environment.reset(ANT_STARTING_X, ANT_STARTING_Y, DIRECTION_RIGHT)
            agent.num_runs += 1
            agent.train_long_memory()

            if score > max_score:
                max_score = score

            if time_steps > max_time_steps:
                max_time_steps = time_steps

            print(
                f"Run #{agent.num_runs} - Score: {score}, Highest Score: {max_score} | "
                f"Time Steps: {time_steps}, Highest Time Steps: {max_time_steps}"
            )


if __name__ == "__main__":
    train()
