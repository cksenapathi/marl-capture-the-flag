import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

game_board = np.ones([20, 20])

num_players = 3

cmap = ListedColormap(['k', 'g', 'r', 'b', 'y', 'w'])

team1_x_pos = np.random.randint(15, 20, 3)
team1_y_pos = np.random.randint(0, 20, 3)

flag1_x_pos = np.random.randint(15, 20, 1)
flag1_y_pos = np.random.randint(0, 20, 1)

team2_x_pos = np.random.randint(0, 5, 3)
team2_y_pos = np.random.randint(0, 20, 3)

flag2_x_pos = np.random.randint(0, 5, 1)
flag2_y_pos = np.random.randint(0, 20, 1)


game_board[team1_x_pos, team1_y_pos] = 2
game_board[team2_x_pos, team2_y_pos] = 3

game_board[flag1_x_pos, flag1_y_pos] = 4
game_board[flag2_x_pos, flag2_y_pos] = 5

plt.matshow(game_board, cmap=cmap)

def create_team_mask(team_x_pos, team_y_pos):
    team_mask = np.zeros_like(game_board)

    for (x, y) in zip(team_x_pos, team_y_pos):
        team_mask[x-1:x+2, y-1:y+2] = 1

    return team_mask    


team1_mask = create_team_mask(team1_x_pos, team1_y_pos)

plt.matshow(game_board * team1_mask, cmap=cmap)
plt.show()


