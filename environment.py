import pygame
import csv
import os

pygame.font.init()

SCORE_FONT = pygame.font.SysFont("arial", 25)

GRID_WIDTH, GRID_HEIGHT = 32, 32
CELL_WIDTH, CELL_HEIGHT = 28, 28
WINDOW_WIDTH = GRID_WIDTH * CELL_WIDTH
WINDOW_HEIGHT = GRID_HEIGHT * CELL_HEIGHT

BACKGROUND_COLOR = (255, 255, 153)
GRID_LINE_COLOR = (187, 187, 131)
TEXT_COLOR = (0, 0, 0)
GRID_LINE_WIDTH = 4

FOOD_PELLET_WIDTH, FOOD_PELLET_HEIGHT = 47, 29
FOOD_PELLET_COLOR = (0, 0, 204)

ANT_WIDTH, ANT_HEIGHT = 20, 25
ANT_STARTING_X = 0 * CELL_WIDTH + (CELL_WIDTH - ANT_WIDTH) / 2
ANT_STARTING_Y = 0 * CELL_HEIGHT + (CELL_HEIGHT - ANT_HEIGHT) / 2
ANT_STARTING_DIRECTION = 270

ANT_SPRITE = pygame.image.load(os.path.join("assets", "ant.png"))
ANT = pygame.transform.scale(ANT_SPRITE, (ANT_WIDTH, ANT_HEIGHT))

FOOD_PELLET_SPRITE = pygame.image.load(
    os.path.join("assets", "food_pellet.png")
)
FOOD_PELLET = pygame.transform.scale(
    FOOD_PELLET_SPRITE, (FOOD_PELLET_WIDTH, FOOD_PELLET_HEIGHT)
)

FPS = 20
MAX_TIME_STEPS = 200


class SantaFeEnvironment:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Santa Fe Trail Simulation")

        self.clock = pygame.time.Clock()
        self.reset(ANT_STARTING_X, ANT_STARTING_Y, ANT_STARTING_DIRECTION)

    def step(self, action):
        self.current_time_step += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.move(action)

        reward = 0
        run = True

        if (self.current_time_step > MAX_TIME_STEPS) or len(
            self.food_pellet_positions
        ) == 0:
            run = False
            reward = -10
            return run, reward, self.score

        current_ant_position = (self.x // CELL_WIDTH, self.y // CELL_HEIGHT)

        if current_ant_position in self.food_pellet_positions:
            self.food_pellet_positions.remove(current_ant_position)
            self.score += 1
            reward = 10

        self.draw_window()
        self.clock.tick(FPS)
        return run, reward, self.score

    def reset(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction

        self.score = 0
        self.current_time_step = 0

        self.food_pellet_positions = self.load_food_pellet_positions(
            os.path.join("assets", "food_pellet_positions.csv")
        )

    def move(self, action):
        pass

    def load_food_pellet_positions(self, filename):
        positions = []

        with open(filename, "r") as file:
            reader = csv.reader(file)

            for row in reader:
                x, y = map(int, row)
                positions.append((x - 1, y - 1))
        return positions

    def draw_food_pellets(self):
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
        self.window.fill(BACKGROUND_COLOR)
        self.draw_grid()
        self.draw_food_pellets()

        self.window.blit(
            pygame.transform.rotate(ANT, self.direction), (self.x, self.y)
        )

        score_text = SCORE_FONT.render(f"Score: {self.score}", True, TEXT_COLOR)
        self.window.blit(score_text, (0, 0))
        pygame.display.update()


def main():
    run = True

    env = SantaFeEnvironment(WINDOW_WIDTH, WINDOW_HEIGHT)

    while run:
        env.clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    env.x += CELL_WIDTH
                    env.direction = 270
                    run, reward, score = env.step(0)
                if event.key == pygame.K_DOWN:
                    env.y += CELL_HEIGHT
                    env.direction = 180
                    run, reward, score = env.step(0)
                if event.key == pygame.K_UP:
                    env.y -= CELL_HEIGHT
                    env.direction = 0
                    run, reward, score = env.step(0)
                if event.key == pygame.K_LEFT:
                    env.x -= CELL_HEIGHT
                    env.direction = 90
                    run, reward, score = env.step(0)

        env.draw_window()


if __name__ == "__main__":
    main()
