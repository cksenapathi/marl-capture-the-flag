import numpy as np
import pygame
import pygame_gui
import os
import time

def dist(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2, axis=-1))

# Player class
class Player(pygame.sprite.Sprite):
    def __init__(self, image_path, pos=None) -> None:
        super().__init__()

        # Sprite stuff
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (50, 60)) # Resize image

        if pos is None:
            pos = np.array([0, 0])

        self.set_pos(pos)
        self.active = True
        self.action = 0.0

    def set_pos(self, pos):
        self.pos = pos

    def set_action(self, action):
        self.action = action

    def get_pos(self):
        return self.pos
    
    def update_pos(self, action, board_dims):
        self.action = action
        self.pos += np.array([np.cos(action), np.sin(action)])

        np.clip(self.pos, a_min=np.zeros_like(board_dims), a_max=board_dims, out=self.pos)

        return self.pos
    
    def set_inactive(self):
        self.active = False

    def set_active(self):
        self.active = True


class Team:
    def __init__(self, flag_path, players=[], flag_pos=None) -> None:
        self.players = players
        self.flag_pos = flag_pos
        # self.inactive_players = []

        # Get Sprite
        self.flag_image = pygame.image.load(flag_path)
        self.flag_image = pygame.transform.scale(self.flag_image, (40, 60))

    def set_random_pos(self, lo_bound, hi_bound):
        for player in self.players:
            player.set_active()
            pos = np.random.uniform(lo_bound, hi_bound)
            player.set_pos(pos)

        self.flag_pos = np.random.uniform(lo_bound, hi_bound)

    def get_pos(self):
        pos_lst = [p.get_pos() if p.active else np.zeros([2,]) for p in self.players]
        pos_lst.append(self.flag_pos)

        return np.asarray(pos_lst)
    
    def apply_action(self, action, board_dims):
        if action is None:
            new_poses = [p.get_pos() for p in self.players]
        else:
            new_poses = [p.update_pos(action[i], board_dims) if 
                        p.active else p.get_pos() for i, p in enumerate(self.players)]
            
        return np.asarray(new_poses)
    
    def num_active_players(self):
        return sum([1 if p.active else 0 for p in self.players])

    def sample_action(self):
        return np.array([np.random.uniform(0, 2*np.pi) for _ in self.players])
    
    def remove_players(self, inactive_players):
        # print("Pre Removal ", self.inactive_players, self.active_players)
        for p in reversed(inactive_players):
            self.players[p].set_inactive()
            self.players[p].set_pos(np.zeros([2,]))
        # print("Post Removal ", self.inactive_players, self.active_players)
            
    def get_action(self):
        return np.array([p.action if p.active else 0.0 for p in self.players])



class Game:
    def __init__(self, team_sprite_path, team_flag_path, T=200, screen_width=800, screen_height=800) -> None:
        self.board_dims = np.array([30, 30])
        self.x_scale = 800 / (self.board_dims[0])
        self.y_scale = 800 / (self.board_dims[0])
        self.xmin = 0
        self.ymin = 0

        self.screen_width = screen_width
        self.screen_height = screen_height

        # Team setup
        self.team1 = Team(team_flag_path[0], [Player(team_sprite_path[0]) for _ in range(3)])
        self.team2 = Team(team_flag_path[1], [Player(team_sprite_path[1]) for _ in range(3)])

        self.team1_bounds = np.array([[0, 0], [30, 10]])
        self.team2_bounds = np.array([[0, 20], [30, 30]])

        self.interaction_radius = 5.0

        self.game_done = False
        self.draw = False

        self.initialized = False

        # Time steps
        self.t = 0
        self.T = T # horizon

    def _check_distances(self):

        if self.team1.num_active_players() == 0 and self.team2.num_active_players() == 0:
            self.game_done = True
            return

        team1_pos = self.team1.get_pos()
        team2_pos = self.team2.get_pos()

        team1_pos, flag1_pos = team1_pos[:-1], team1_pos[-1]
        team2_pos, flag2_pos = team2_pos[:-1], team2_pos[-1]

        team1_flag_score, team2_flag_score = 0, 0
        team1_player_score = np.zeros([len(team1_pos)])
        team2_player_score = np.zeros([len(team2_pos)])

        for pos in team1_pos:
            team1_flag_score += (1.5*(dist(pos, flag1_pos))*(dist(pos, flag1_pos)<self.interaction_radius))
            team2_flag_score -= ((dist(pos, flag2_pos)) * (dist(pos, flag2_pos)<self.interaction_radius))

            team1_player_score += (1.5*(dist(pos, team1_pos)) * (dist(pos, team1_pos)<self.interaction_radius))
            team2_player_score -= ((dist(pos, team2_pos)) * (dist(pos, team2_pos)<self.interaction_radius))

        
        for pos in team2_pos:
            team2_flag_score += (1.5*(dist(pos, flag2_pos))*(dist(pos, flag2_pos)<self.interaction_radius))
            team1_flag_score -= ((dist(pos, flag1_pos)) * (dist(pos, flag1_pos)<self.interaction_radius))

            team2_player_score += (1.5*(dist(pos, team2_pos)) * (dist(pos, team2_pos)<self.interaction_radius))
            team1_player_score -= ((dist(pos, team1_pos)) * (dist(pos, team1_pos)<self.interaction_radius))

        flag1_cap = team1_flag_score < 0
        flag2_cap = team2_flag_score < 0
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
        
        t1_inactive_players = [i for i,s in enumerate(team1_player_score) if s < 0]
        t2_inactive_players = [i for i,s in enumerate(team2_player_score) if s < 0]

        self.team1_reward += (3*len(t2_inactive_players) - 3*len(t1_inactive_players))
        self.team2_reward += (3*len(t1_inactive_players) - 3*len(t2_inactive_players))
        
        self.team1.remove_players(t1_inactive_players)
        self.team2.remove_players(t2_inactive_players)

    
    def render(self):
        if not self.initialized:
            raise ValueError("Environment not initialized. Call reset() before calling render().")
        
        time.sleep(0.075)

        # Clear screen
        self.screen.fill((209, 255, 214))

        # Draw line and buffer zone
        # Create a surface for the transparent line
        line_surface = pygame.Surface((self.screen.get_width(), 5), pygame.SRCALPHA)  # 5 pixels tall line with transparency
        line_surface.fill((0, 0, 0, 0))  # Transparent background

        # Draw the line on the transparent surface
        mid_y = self.screen.get_height() // 2  # Middle of the screen (Y-coordinate)
        pygame.draw.line(line_surface, (0, 0, 0, 120), (0, 0), (self.screen.get_width(), 0), 5)  # Line with alpha

        # Blit the transparent line surface onto the main screen at the correct position
        self.screen.blit(line_surface, (0, mid_y - 2))  # Center the line vertically

        # Draw a shaded region (buffer zone) in the middle of the screen
        buffer_zone_height = 10 * self.y_scale # Height of the buffer zone
        buffer_zone_width = self.screen.get_width()
        buffer_zone_rect = pygame.Rect(0, mid_y - buffer_zone_height // 2, buffer_zone_width, buffer_zone_height)

        # Add some transparency to the buffer zone by using a surface
        buffer_zone_surface = pygame.Surface((buffer_zone_width, buffer_zone_height))
        buffer_zone_surface.set_alpha(40)  # Set alpha for transparency
        buffer_zone_surface.fill((0, 0, 0))  # Fill with blue color (can change)

        # Blit the buffer zone surface to the screen
        self.screen.blit(buffer_zone_surface, (0, mid_y - buffer_zone_height // 2))
        for player in self.team1.players:
            if player.active:
                pos = player.get_pos()
                grid_pos = self._grid_to_screen(pos)

                self.screen.blit(player.image, grid_pos)

        for player in self.team2.players:
            if player.active:
                pos = player.get_pos()
                grid_pos = self._grid_to_screen(pos)

                self.screen.blit(player.image, grid_pos)

        # Draw flags
        flag1_pos = self.team1.flag_pos
        flag2_pos = self.team2.flag_pos
        flag1_screen_pos = self._grid_to_screen(flag1_pos)
        flag2_screen_pos = self._grid_to_screen(flag2_pos)

        self.screen.blit(self.team1.flag_image, flag1_screen_pos)
        self.screen.blit(self.team2.flag_image, flag2_screen_pos)

        pygame.display.flip()


    # Move one unit step in the direction specified
    def step(self, team1_action, team2_action):
        if not self.initialized:
            raise ValueError("Environment not initialized. Call reset() before calling step().")
        
        if self.t >= self.T:
            return True, self.team1.get_pos(), self.team2.get_pos(), self.team1.get_action(), self.team2.get_action(), self.team1_reward, self.team2_reward
        
        print("\n\n",self.t)

        self.team1_reward = -0.1
        self.team2_reward = -0.1
        t1_pos = self.team1.apply_action(team1_action, self.board_dims)
        t2_pos = self.team2.apply_action(team2_action, self.board_dims)
        self._check_distances()

        self.t += 1

        if self.t >= self.T:
            self.game_done = True
            self.winner = -1

        done = self.game_done
        # pos_obs = np.concat([self.team1.get_pos(), self.team2.get_pos()]).flatten()

        return done, t1_pos, t2_pos, self.team1.get_action(), self.team2.get_action(), self.team1_reward, self.team2_reward


    def _grid_to_screen(self, pos):
        # Convert grid position to screen coordinates
        screen_x = self.x_scale * (pos[0] - self.xmin)
        screen_y = 800 - self.y_scale * (pos[1] - self.ymin)  # Invert Y to match Pygame screen
        return (screen_x, screen_y)   


    @property
    # Return state
    def _state(self):
        pass


    @property
    def _is_terminal(self):
        self.initialized = False

        # Check if all agents are captured 
        if len(self.team1.inactive_players) == 3 or len(self.team2.inactive_players) == 3:
            return True
        
        # Check if flag is captured
        # TODO

        # Check if max time steps reached
        if self.t == self.T:
            return True


    def reset(self):
        self.team1_reward = 0
        self.team2_reward = 0
        self.game_done = False
        self.t = 0

        self.initialized = True

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.manager = pygame_gui.UIManager((self.screen_width, self.screen_height))

    
        self.team1.set_random_pos(self.team1_bounds[0], self.team1_bounds[1])
        self.team2.set_random_pos(self.team2_bounds[0], self.team2_bounds[1])
        
        return self.team1.get_pos(), self.team2.get_pos()
