# import random
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
# from collections import deque
# from game import SnakeGameAI, Point, BLOCK_SIZE
# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# MAX_MEMORY = 1_000
# # view_size = 7
# # input_shape = (view_size * view_size + 2,)
# BATCH_SIZE = 32
# input_shape = (8,)
#
#
# class Agent:
#
#     def __init__(self):
#         self.n_games = 0
#         self.epsilon = 80  # randomness
#         self.gamma = 0.9  # discount rate
#         self.memory = deque(maxlen=MAX_MEMORY)
#         self.model = keras.Sequential(
#             [
#                 keras.Input(shape=input_shape),
#                 layers.Dense(128, activation="relu"),
#                 layers.Dense(128, activation="relu"),
#                 layers.Dense(3, activation="softmax"),
#             ]
#             , name="model")
#         self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1),
#                            loss=keras.losses.Huber(), metrics=['accuracy'])
#
#     def train_step(self, old_state, reward, done, action, new_state):
#         reshaped_old_state = np.reshape(old_state, (1, *input_shape))
#         Q = self.model.predict(reshaped_old_state, verbose=0)
#         target = Q.copy()
#         new_Q = reward
#         if not done:
#             new_Q = reward + self.gamma * np.max(
#                 self.model.predict(np.reshape(new_state, (1, *input_shape)), verbose=0))
#         target[0][np.argmax(action)] = new_Q
#         data = (reshaped_old_state, tf.constant(target))
#         self.model.train_step(data)
#         self.memory.append((old_state, action, reward, new_state, done))  # popleft if MAX_MEMORY is reached
#
#     def long_train(self):
#         states, actions, rewards, next_states, dones = zip(*self.memory)
#         states = np.array(states)
#         next_states = np.array(next_states)
#         Qs = self.model(states)
#         Qs = np.array(Qs)
#         targets = Qs.copy()
#         new_Qs = np.array(rewards)
#         for idx in range(len(dones)):
#             if not dones[idx]:
#                 new_Qs[idx] = rewards[idx] + self.gamma * np.max(
#                     self.model(np.reshape(next_states[idx], (1, *input_shape))))
#             targets[idx][np.argmax(actions[idx])] = new_Qs[idx]
#         self.model.fit(states, targets, batch_size=BATCH_SIZE, verbose=0)
#
#     def get_action(self, state_reshaped):
#         # random moves: tradeoff exploration / exploitation
#         epsilon = self.epsilon - self.n_games
#         final_move = [0, 0, 0]
#         if random.randint(0, self.epsilon) < epsilon:
#             move = random.randint(0, 2)
#             final_move[move] = 1
#         else:
#             prediction = self.model(state_reshaped, training=True)
#             move = np.argmax(prediction)
#             final_move[move] = 1
#         return final_move
#
#
# # def get_state(game):
# #     state = np.zeros(input_shape)
# #     head = game.head
# #     for i in range(view_size):
# #         for j in range(view_size):
# #             pt = Point(head.x + (j - view_size // 2) * BLOCK_SIZE, head.y + (i - view_size // 2) * BLOCK_SIZE)
# #             if game.is_collision(pt) or (i == view_size // 2 and j == view_size // 2):
# #                 state[i * view_size + j] = 0
# #             else:
# #                 state[i * view_size + j] = 1
# #     state[-2] = (game.food.x - head.x) // BLOCK_SIZE
# #     state[-1] = (game.food.y - head.y) // BLOCK_SIZE
# #     return
#
# def get_state(game):
#     state = np.zeros(input_shape)
#     head = game.head
#     heads = [Point(head.x + BLOCK_SIZE, head.y), Point(head.x - BLOCK_SIZE, head.y),
#              Point(head.x, head.y + BLOCK_SIZE), Point(head.x, head.y - BLOCK_SIZE)]
#     for i, pt in enumerate(heads):
#         state[i] = 1 if game.is_collision(pt) else 0
#     food_dir_bools = [head.x > game.food.x, head.x < game.food.x, head.y > game.food.y, head.y < game.food.y]
#     for i, food_dir_bool in enumerate(food_dir_bools):
#         state[i + 4] = 1 if food_dir_bool else 0
#     # for i in range(view_size):
#     #     for j in range(view_size):
#     #         pt = Point(head.x + (j - view_size // 2) * BLOCK_SIZE, head.y + (i - view_size // 2) * BLOCK_SIZE)
#     #         if game.is_collision(pt) or (i == view_size // 2 and j == view_size // 2):
#     #             state[i * view_size + j] = 0
#     #         else:
#     #             state[i * view_size + j] = 1
#     # state[-2] = (game.food.x - head.x) // BLOCK_SIZE
#     # state[-1] = (game.food.y - head.y) // BLOCK_SIZE
#     return state
#
#
# def train():
#     record = 0
#     agent = Agent()
#     game = SnakeGameAI()
#     while True:
#         # get old state
#         state = get_state(game)
#         # get move
#         final_move = agent.get_action(np.reshape(state, (1, state.size)))
#
#         # perform move and get new state
#         # reward, done, score = game.play_step(final_move, agent.n_games + 1, agent.epsilon)
#         reward, done, score = game.play_step(final_move)
#
#         new_state = get_state(game)
#
#         # train-step
#         agent.train_step(state, reward, done, final_move, new_state)
#
#         if done:
#             agent.n_games += 1
#             agent.long_train()
#             game.reset()
#
#             if score > record:
#                 record = score
#                 agent.model.save("my_model.h5")
#
#             print('Game', agent.n_games, 'Score', score, 'Record:', record)
#
#
# if __name__ == '__main__':
#     train()
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer

# from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.1
view_size = 3
input_shape = view_size * view_size + 2


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.99  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(input_shape, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    @staticmethod
    def get_state(game):
        vision_matrix = np.zeros((view_size, view_size))
        head = game.head
        for i in range(view_size):
            for j in range(view_size):
                pt = Point(head.x + (j - view_size // 2) * BLOCK_SIZE, head.y + (i - view_size // 2) * BLOCK_SIZE)
                if game.is_collision(pt) or (i == view_size // 2 and j == view_size // 2):
                    vision_matrix[i,j] = -1
                else:
                    vision_matrix[i,j] = 1
        if game.direction == Direction.RIGHT:
            vision_matrix = np.rot90(vision_matrix)
        elif game.direction == Direction.DOWN:
            vision_matrix = np.rot90(vision_matrix, k=2)
        elif game.direction == Direction.LEFT:
            vision_matrix = np.rot90(vision_matrix, k=3)
        state = np.zeros(input_shape)
        state[:(view_size*view_size)] = vision_matrix.flatten()
        state[-2] = (game.food.x - head.x)//BLOCK_SIZE
        state[-1] = (game.food.y - head.y)//BLOCK_SIZE
        return state

    # def get_state(self, game):
    #     head = game.snake[0]
    #     point_l = Point(head.x - 20, head.y)
    #     point_r = Point(head.x + 20, head.y)
    #     point_u = Point(head.x, head.y - 20)
    #     point_d = Point(head.x, head.y + 20)
    #
    #     dir_l = game.direction == Direction.LEFT
    #     dir_r = game.direction == Direction.RIGHT
    #     dir_u = game.direction == Direction.UP
    #     dir_d = game.direction == Direction.DOWN
    #
    #     state = [
    #         # Danger straight
    #         (dir_r and game.is_collision(point_r)) or
    #         (dir_l and game.is_collision(point_l)) or
    #         (dir_u and game.is_collision(point_u)) or
    #         (dir_d and game.is_collision(point_d)),
    #
    #         # Danger right
    #         (dir_u and game.is_collision(point_r)) or
    #         (dir_d and game.is_collision(point_l)) or
    #         (dir_l and game.is_collision(point_u)) or
    #         (dir_r and game.is_collision(point_d)),
    #
    #         # Danger left
    #         (dir_d and game.is_collision(point_r)) or
    #         (dir_u and game.is_collision(point_l)) or
    #         (dir_r and game.is_collision(point_u)) or
    #         (dir_l and game.is_collision(point_d)),
    #
    #         # Move direction
    #         dir_l,
    #         dir_r,
    #         dir_u,
    #         dir_d,
    #
    #         # Food location
    #         game.food.x < game.head.x,  # food left
    #         game.food.x > game.head.x,  # food right
    #         game.food.y < game.head.y,  # food up
    #         game.food.y > game.head.y  # food down
    #     ]
    #
    #     return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 500 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 1250) < self.epsilon:
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
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # plot_scores.append(score)
            # total_score += score
            # mean_score = total_score / agent.n_games
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
