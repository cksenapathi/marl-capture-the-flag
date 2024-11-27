import numpy as np

class Player:
    def __init__(self, pos=None) -> None:
        self.pos = pos
        self.action = None

    def set_pos(self, pos):
        self.pos = pos

    def get_pos(self):
        return self.pos
    
    def update_pos(self, action, board_dims):
        self.action = action
        self.pos += np.array([np.cos(action), np.sin(action)])

        np.clip(self.pos, a_min=np.zeros_like(board_dims), a_max=board_dims, out=self.pos)

        return self.pos


class Team:
    def __init__(self, players=[], flag_pos=None) -> None:
        self.active_players = players
        self.flag_pos = flag_pos
        self.inactive_players = []

    def set_random_pos(self, lo_bound, hi_bound):
        for player in self.active_players:
            pos = np.random.uniform(lo_bound, hi_bound)
            player.set_pos(pos)

        self.flag_pos = np.random.uniform(lo_bound, hi_bound)

    def get_pos(self):
        pos_lst = [p.get_pos() for p in self.active_players]
        pos_lst.append(self.flag_pos)
        return np.asarray(pos_lst)
    
    def apply_action(self, action, board_dims):
        new_poses = []
        for i, p in enumerate(self.active_players):
            new_pos = p.update_pos(action[i], board_dims)
            new_poses.append(new_pos)

    def sample_action(self):
        return np.array([np.random.uniform(0, 2*np.pi) for _ in self.active_players])
    
    def remove_players(self, inactive_players):
        print("Pre Removal ", self.inactive_players, self.active_players)
        for p in inactive_players:
            self.inactive_players.append(self.active_players[p])
        for p in self.inactive_players:
            self.active_players.remove(p)
        print("Post Removal ", self.inactive_players, self.active_players)


class Game:
    def __init__(self) -> None:
        self.team1 = Team([Player(), Player(), Player()])
        self.team2 = Team([Player(), Player(), Player()])

        self.board_dims = np.array([30, 30])

        self.team1_bounds = np.array([[0, 0],[30,10]])
        self.team2_bounds = np.array([[0, 20],[30,30]])

        self.player_interaction_radius = 5.0

        self.game_done = False
        self.draw = False

    
    def _check_distances(self):
        team1_pos = self.team1.get_pos()
        team2_pos = self.team2.get_pos()

        # print(team1_pos)
        # print(team2_pos)

        # A[i,j] gives the distance b/w Player i on team 1
        # and Player j on team 2
        # The 4th element of each dimension is the team flag
        inter_team_dist = np.sqrt(
            ((team1_pos[:, None] - team2_pos[None, :, :])**2).sum(-1)
        )

        inter_team_dist *= inter_team_dist<=self.player_interaction_radius

        # Distance between players on Team 1
        team1_dist = np.sqrt(
            ((team1_pos[:, None] - team1_pos[None, :, :])**2).sum(-1)
        )

        team1_dist *= team1_dist<=self.player_interaction_radius


        team2_dist = np.sqrt(
            ((team2_pos[:, None] - team2_pos[None, :, :])**2).sum(-1)
        )

        team2_dist *= team2_dist<=self.player_interaction_radius


        team1_score = (1.5*team1_dist - inter_team_dist).sum(1)
        team2_score = (1.5*team2_dist - inter_team_dist).sum(0)

        print(team1_score, team2_score)

        flag1_cap = team1_score[-1] < 0
        flag2_cap = team2_score[-1] < 0
        self.game_done = flag1_cap or flag2_cap
        self.draw = flag2_cap and flag1_cap

        if self.game_done:
            if not self.draw:
                if flag2_cap:
                    self.winner = 1
                    self.team1_reward += 10
                    self.team2_reward -= 10
                else:
                    self.winner = 2
                    self.team2_reward += 10
                    self.team1_reward -= 10
            else:
                self.winner = None
                self.team1_reward -= 20
                self.team2_reward -= 20
            return
        
        t1_inactive_players = [i for i,s in enumerate(team1_score[:-1]) if s < 0]
        t2_inactive_players = [i for i,s in enumerate(team2_score[:-1]) if s < 0]

        self.team1_reward += (3*len(t2_inactive_players) - 3*len(t1_inactive_players))
        self.team2_reward += (3*len(t1_inactive_players) - 3*len(t2_inactive_players))
        
        self.team1.remove_players(t1_inactive_players)
        self.team2.remove_players(t2_inactive_players)



    def step(self, team1_action, team2_action):
        self.team1_reward -= 0.1
        self.team2_reward -= 0.1
        self.team1.apply_action(team1_action, self.board_dims)
        self.team2.apply_action(team2_action, self.board_dims)
        self._check_distances()


    def reset_board(self):
        self.team1_reward = 0
        self.team2_reward = 0
        self.game_done = False
        self.team1.set_random_pos(self.team1_bounds[0], self.team1_bounds[1])
        self.team2.set_random_pos(self.team2_bounds[0], self.team2_bounds[1])


game = Game()
game.reset_board()

for _ in range(100):
    team_1_act = game.team1.sample_action()
    team_2_act = game.team2.sample_action()

    game.step(team_1_act, team_2_act)
