import numpy as np

class Player:
    def __init__(self, pos=None) -> None:
        self.pos = pos
        self.action = None

    def set_pos(self, pos):
        self.pos = pos

    def get_pos(self):
        return self.pos
    
    def update_pos(self, action):
        self.pos += np.array([np.cos(action), np.sin(action)])


class Team:
    def __init__(self, players=[], flag_pos=None) -> None:
        self.players = players
        self.flag_pos = flag_pos
        self.inactive_players = []

    def set_random_pos(self, lo_bound, hi_bound):
        for player in self.players:
            pos = np.random.uniform(lo_bound, hi_bound)
            player.set_pos(pos)

        self.flag_pos = np.random.uniform(lo_bound, hi_bound)

    def get_pos(self):
        pos_lst = [p.get_pos() for p in self.players]
        pos_lst.append(self.flag_pos)
        return np.asarray(pos_lst)
    
    def apply_action(self, action):
        for i, p in enumerate(self.players):
            p.update_pos(action[i])



class Game:
    def __init__(self) -> None:
        self.team1 = Team([Player(), Player(), Player()])
        self.team2 = Team([Player(), Player(), Player()])

        self.board_dims = np.array([30, 30])

        self.team1_bounds = np.array([[0, 0],[30,10]])
        self.team2_bounds = np.array([[0, 20],[30,30]])

        self.player_interaction_radius = 20.0

    
    def check_distances(self):
        team1_pos = self.team1.get_pos()
        team2_pos = self.team2.get_pos()

        print(team1_pos)
        print(team2_pos)

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


        team1_score = (1.5*team1_dist - inter_team_dist).sum(0)
        team2_score = (1.5*team2_dist - inter_team_dist).sum(1)

        # Check if player is eliminated
            # negative score for player implies death
        # Check if game is over
            # negative score for flag implies capture


    def step(self, team1_action, team2_action):
        self.team1.apply_action(team1_action)
        self.team2.apply_action(team2_action)


    def reset_board(self):
        self.team1.set_random_pos(self.team1_bounds[0], self.team1_bounds[1])
        self.team2.set_random_pos(self.team2_bounds[0], self.team2_bounds[1])


game = Game()
game.reset_board()
game.check_distances()