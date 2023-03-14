import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

"""
this whole was taken from https://github.com/patrickloeber/python-fun/blob/master/snake-pygame/snake_game.py
with few extra add-ons I made. didn't want to code the game, but to make the RL part
"""
pygame.init()
font = pygame.font.Font('arial.ttf', 25)


# font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'game_over'))

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 50


class SnakeGame:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.direction = None
        self.head = None
        self.snake = None
        self.score = None
        self.food = None
        self.restart()

    def restart(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    # made it to make the snake unable to rotate himself 180 degrees in 1 move
    @staticmethod
    def opposite_move(direction, as_pygame=True):
        if direction == Direction.LEFT:
            return pygame.K_RIGHT if as_pygame else Direction.RIGHT
        elif direction == Direction.RIGHT:
            return pygame.K_LEFT if as_pygame else Direction.LEFT
        elif direction == Direction.UP:
            return pygame.K_DOWN if as_pygame else Direction.DOWN
        else:
            return pygame.K_UP if as_pygame else Direction.UP

    @staticmethod
    def get_right_dir(direction):
        if direction == Direction.LEFT:
            return Direction.UP
        elif direction == Direction.RIGHT:
            return Direction.DOWN
        elif direction == Direction.UP:
            return Direction.RIGHT
        else:
            return Direction.LEFT

    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == self.opposite_move(self.direction):
                    break
                elif event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN

        # 2. move
        food_loc = self.food
        self._move(self.direction)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        reward = 0
        if self.head == self.food:
            reward = 10
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def _is_collision(self):
        # hits boundary
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        # hits itself
        if self.head in self.snake[1:]:
            return True
        return False

    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    """
    state = [danger_straight_dist , danger_right_dist , danger_left_dist , \
            dir_right_bool , dir_left_bool , dir_up_bool , dir_down_bool ,\
            food_right_bool , food_left_bool , food_up_bool , food_down_bool]
    """

    def get_State(self):
        direction = self.direction
        direction_right = self.get_right_dir(direction)
        direction_left = self.opposite_move(direction_right)
        dirs = [direction, direction_right, direction_left]
        collisions = [1] * 3
        for i, dir in enumerate(dirs):
            if not self._is_collision():
                collisions[i] = 0
        direction_bools = [0] * 4
        direction_bools[direction.value - 1] = 1
        food_bools = [0] * 4
        if self.head.x < self.food.x:
            food_bools[0] = 1
        elif self.head.x > self.food.x:
            food_bools[1] = 1
        if self.head.y < self.food.y:
            food_bools[2] = 1
        elif self.head.y > self.food.y:
            food_bools[3] = 1
        return np.array([*collisions, *direction_bools, *food_bools], dtype=int)

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

# if __name__ == '__main__':
#     game = SnakeGame()
#     # game loop
#     while True:
#         game_over, score = game.play_step()[1:]
#         if game_over:
#             break
#
#     print('Final Score', score)
#     pygame.quit()
