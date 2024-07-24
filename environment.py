import pygame
import csv
import os
import numpy as np

# Initialize fonts
pygame.font.init()

# Font settings for score and time steps
SCORE_FONT = pygame.font.SysFont("arial", 20)
TIME_STEPS_FONT = pygame.font.SysFont("arial", 20)

# Grid and window settings
GRID_WIDTH, GRID_HEIGHT = 32, 32
GRID_LINE_WIDTH = 4
CELL_WIDTH, CELL_HEIGHT = 28, 28
WINDOW_WIDTH = GRID_WIDTH * CELL_WIDTH
WINDOW_HEIGHT = GRID_HEIGHT * CELL_HEIGHT

# Colors
BACKGROUND_COLOR = (255, 255, 153)
GRID_LINE_COLOR = (187, 187, 131)
TEXT_COLOR = (0, 0, 0)
FOOD_PELLET_COLOR = (0, 0, 204)

# Food pellet settings
FOOD_PELLET_WIDTH, FOOD_PELLET_HEIGHT = 47, 29
FOOD_PELLET_SPRITE = pygame.image.load(
    os.path.join("assets", "food_pellet.png")
)
FOOD_PELLET = pygame.transform.scale(
    FOOD_PELLET_SPRITE, (FOOD_PELLET_WIDTH, FOOD_PELLET_HEIGHT)
)

# Ant settings
ANT_WIDTH, ANT_HEIGHT = 20, 25
ANT_STARTING_X = (
    0 * CELL_WIDTH + (CELL_WIDTH - ANT_WIDTH) / 2 + GRID_LINE_WIDTH / 4
)
ANT_STARTING_Y = (
    0 * CELL_HEIGHT + (CELL_HEIGHT - ANT_HEIGHT) / 2 + GRID_LINE_WIDTH / 4
)
ANT_SPRITE = pygame.image.load(os.path.join("assets", "ant.png"))
ANT = pygame.transform.scale(ANT_SPRITE, (ANT_WIDTH, ANT_HEIGHT))

# Directions
DIRECTION_RIGHT = 270
DIRECTION_LEFT = 90
DIRECTION_UP = 0
DIRECTION_DOWN = 180

# Simulation settings
FPS = 60
MAX_TIME_STEPS = 150


class SantaFeEnvironment:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Initialize Pygame window and clock
        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Santa Fe Trail Simulation")

        self.clock = pygame.time.Clock()
        self.reset(ANT_STARTING_X, ANT_STARTING_Y, DIRECTION_RIGHT)

    def step(self, action):
        # Increment time steps
        self.current_time_steps += 1
        self.remaining_time_steps -= 1

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Move the ant based on the action
        self.move(action)

        reward = 0
        run = True

        # Check if all the food pellets are collected
        if len(self.food_pellet_positions) == 0:
            run = False
            reward = (self.score / self.current_time_steps) * 100
            return run, reward, self.score, self.current_time_steps - 1

        # Check if the max time steps are reached
        if self.current_time_steps > MAX_TIME_STEPS:
            run = False
            reward = -100
            return run, reward, self.score, self.current_time_steps - 1

        # Calculate current ant position
        current_ant_position = (
            int(self.x // CELL_WIDTH),
            int(self.y // CELL_HEIGHT),
        )

        # Check if the ant found a food pellet
        if current_ant_position in self.food_pellet_positions:
            self.food_pellet_positions.remove(current_ant_position)
            self.score += 1
            reward = (self.score / self.current_time_steps) * 10

        # Update the window
        self.draw_window()
        self.clock.tick(FPS)
        return run, reward, self.score, self.current_time_steps

    def reset(self, x, y, direction):
        # Reset ant position, direction, score, and time steps
        self.x = x
        self.y = y
        self.direction = direction

        self.score = 0
        self.current_time_steps = 0
        self.remaining_time_steps = MAX_TIME_STEPS

        # Load food pellet positions
        self.food_pellet_positions = self.load_food_pellet_positions(
            os.path.join("assets", "food_pellet_positions.csv")
        )

        # Update the window
        self.draw_window()

    def move(self, action):
        # Action mapping ([Straight, Right, Left])
        clockwise = [
            DIRECTION_RIGHT,
            DIRECTION_DOWN,
            DIRECTION_LEFT,
            DIRECTION_UP,
        ]
        index = clockwise.index(self.direction)

        # Determine new direction based on action
        if np.array_equal(action, [1, 0, 0]):
            self.direction = clockwise[index]  # Straight
        elif np.array_equal(action, [0, 1, 0]):
            self.direction = clockwise[(index + 1) % 4]  # Right turn
        else:
            self.direction = clockwise[(index - 1) % 4]  # Left turn

        # Update ant position based on direction
        if self.direction == DIRECTION_RIGHT:
            self.x += CELL_WIDTH
        elif self.direction == DIRECTION_DOWN:
            self.y += CELL_HEIGHT
        elif self.direction == DIRECTION_LEFT:
            self.x -= CELL_WIDTH
        elif self.direction == DIRECTION_UP:
            self.y -= CELL_HEIGHT

        # Handle logic for wrapping around the grid
        if self.x >= self.width:
            self.x = 0
        elif self.x < 0:
            self.x = self.width - CELL_WIDTH

        if self.y >= self.height:
            self.y = 0
        elif self.y < 0:
            self.y = self.height - CELL_HEIGHT

    def load_food_pellet_positions(self, filename):
        # Load food pellet positions from CSV file
        positions = []

        with open(filename, "r") as file:
            reader = csv.reader(file)

            for row in reader:
                x, y = map(int, row)
                positions.append((x - 1, y - 1))
        return positions

    def draw_food_pellets(self):
        # Draw food pellets on the window
        for position in self.food_pellet_positions:
            pixel_x = (
                (position[0]) * CELL_WIDTH
                + (CELL_WIDTH - FOOD_PELLET_WIDTH) / 2
                + GRID_LINE_WIDTH / 4
            )
            pixel_y = (
                (position[1]) * CELL_HEIGHT
                + (CELL_HEIGHT - FOOD_PELLET_HEIGHT) / 2
                + GRID_LINE_WIDTH / 4
            )
            self.window.blit(FOOD_PELLET, (pixel_x, pixel_y))

    def draw_grid(self):
        # Draw the grid lines on the window
        for x in range(CELL_WIDTH, WINDOW_WIDTH, CELL_WIDTH):
            pygame.draw.line(
                self.window,
                GRID_LINE_COLOR,
                (x, 0),
                (x, WINDOW_HEIGHT),
                GRID_LINE_WIDTH,
            )

        for y in range(CELL_HEIGHT, WINDOW_HEIGHT, CELL_HEIGHT):
            pygame.draw.line(
                self.window,
                GRID_LINE_COLOR,
                (0, y),
                (WINDOW_WIDTH, y),
                GRID_LINE_WIDTH,
            )

    def draw_window(self):
        # Fill the background
        self.window.fill(BACKGROUND_COLOR)
        self.draw_grid()
        self.draw_food_pellets()

        # Draw the ant
        self.window.blit(
            pygame.transform.rotate(ANT, self.direction), (self.x, self.y)
        )

        # Draw the score and remaining time steps
        score_text = SCORE_FONT.render(f"Score: {self.score}", True, TEXT_COLOR)
        time_steps_text = TIME_STEPS_FONT.render(
            f"Remaining Time Steps: {self.remaining_time_steps}",
            True,
            TEXT_COLOR,
        )
        self.window.blit(score_text, (0, 0))
        self.window.blit(time_steps_text, (652, 0))

        # Update the display
        pygame.display.update()
