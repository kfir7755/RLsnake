import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from game import *

BATCH_SIZE = 32


# import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='good_model'):
        path = r"C:\Users\kfir\PycharmProjects\RLsnake"
        model_folder_path = path
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class agent:
    def __init__(self, decisions_model=None, games_played=0, lr=0.01, gamma=0.9):
        if decisions_model is not None:
            self.decisions_model = decisions_model
        else:
            self.decisions_model = Net(11, 256, 3)
        self.games_played = games_played
        self.snakeGame = SnakeGame()
        self.epsilon = 70 - self.games_played
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.optimizer = optim.SGD(self.decisions_model.parameters(), lr=self.lr)
        self.memory = []

    def make_move(self, state):
        direction = self.snakeGame.direction
        direction_right = self.snakeGame.get_right_dir(direction)
        direction_left = self.snakeGame.opposite_move(direction_right, as_pygame=False)
        number_for_random_move = np.random.randint(0, high=200)
        Q = self.decisions_model(state)
        if number_for_random_move < self.epsilon:
            dir_arg = np.random.randint(0, high=2)

        else:
            dir_arg = np.argmax(Q.detach().numpy())

        if dir_arg == 0:
            move_dir = direction
        if dir_arg == 1:
            move_dir = direction_right
        else:
            move_dir = direction_left
        self.snakeGame.direction = move_dir
        reward, game_over, score = self.snakeGame.play_step()
        return dir_arg, reward, game_over, score


agent = agent()
while True:
    record = 0
    state = torch.tensor(agent.snakeGame.get_State()).type(torch.float)
    idx, reward, game_over, score = agent.make_move(state)
    if not game_over:
        new_state = torch.tensor(agent.snakeGame.get_State()).type(torch.float)
        to_memorize = Transition(state.detach().numpy(), idx, new_state.detach().numpy(), reward, game_over)
        agent.memory.append(to_memorize)
        if score > record:
            record = score
            agent.decisions_model.save()

    if game_over:
        if len(agent.memory) > BATCH_SIZE:
            mini_sample = random.sample(agent.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = agent.memory

        states, indices, next_states, rewards, dones = zip(*mini_sample)
        states = torch.tensor(np.array(states), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        Qs = agent.decisions_model(states)
        targets = Qs.clone()
        targets = targets.detach().numpy()
        Qs_new = rewards + agent.gamma * np.max(agent.decisions_model(next_states).detach().numpy(), axis=1)
        targets[:, indices] = Qs_new
        targets = torch.tensor(targets, dtype=torch.float)
        Qs_new = torch.tensor(Qs_new, dtype=torch.float)
        agent.optimizer.zero_grad()
        loss = agent.criterion(targets, Qs)
        loss.backward()
        agent.optimizer.step()
        agent.memory = []
        agent.games_played += 1
        print(agent.games_played)
        agent.epsilon = 100 - agent.games_played
        agent.snakeGame.restart()
