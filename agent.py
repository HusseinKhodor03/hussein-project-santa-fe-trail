import torch
import numpy as np
from environment import (
    SantaFeEnvironment,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    ANT_STARTING_X,
    ANT_STARTING_Y,
    DIRECTION_RIGHT,
)
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001


class Agent:
    def __init__(self):
        self.num_runs = 0
        self.epsilon = 0
        self.gamma = 0
        self.memory = deque(maxlen=MAX_MEMORY)

    def get_state(self, environment):
        pass

    def remember(self, state, action, reward, next_state, run):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self, state, action, reward, next_state, run):
        pass

    def get_action(self, state):
        pass


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
