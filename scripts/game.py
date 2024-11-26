import numpy as np
import pygame
import pygame_gui
import os

# Player class
class Player(pygame.sprite.Sprite):
    def __init__(self, image_path, pos=None) -> None:
        super().__init__()

        # Sprite stuff
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (50, 60)) # Resize image

        # Agent stuff
        if pos is None:
            pos = np.array([0, 0])

        self.set_pos(pos)

    def set_pos(self, pos):
        self.pos = pos

    def get_pos(self):
        return self.pos
    
    def step(self, action):
        self.pos += np.array([np.cos(action), np.sin(action)])
        self.set_pos(self.pos) # Update grid position

class Team:
    def __init__(self, flag_path, players=[], flag_pos=None) -> None:
        self.players = players
        self.flag_pos = flag_pos
        self.inactive_players = []

        # Get Sprite
        self.flag_image = pygame.image.load(flag_path)
        self.flag_image = pygame.transform.scale(self.flag_image, (40, 60))

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
    def __init__(self, team_sprite_path, team_flag_path, screen_width=800, screen_height=800) -> None:
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.manager = pygame_gui.UIManager((screen_width, screen_height))

        self.board_dims = np.array([30, 30])
        self.x_scale = 800 / (self.board_dims[0])
        self.y_scale = 800 / (self.board_dims[0])
        self.xmin = 0
        self.ymin = 0

        # Team setup
        self.team1 = Team(team_flag_path[0], [Player(team_sprite_path[0]) for _ in range(3)])
        self.team2 = Team(team_flag_path[1], [Player(team_sprite_path[1]) for _ in range(3)])

        self.team1_bounds = np.array([[0, 0], [30, 10]])
        self.team2_bounds = np.array([[0, 20], [30, 30]])

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

    def render(self):
        # Clear screen
        self.screen.fill((209, 255, 214))

        # Draw line and buffer zone
        # Create a surface for the transparent line
        line_surface = pygame.Surface((self.screen.get_width(), 5), pygame.SRCALPHA)  # 5 pixels tall line with transparency
        line_surface.fill((0, 0, 0, 0))  # Transparent background

        # Draw the line on the transparent surface
        mid_y = self.screen.get_height() // 2  # Middle of the screen (Y-coordinate)
        pygame.draw.line(line_surface, (0, 0, 0, 150), (0, 0), (self.screen.get_width(), 0), 5)  # Line with alpha

        # Blit the transparent line surface onto the main screen at the correct position
        self.screen.blit(line_surface, (0, mid_y - 2))  # Center the line vertically

        # Draw a shaded region (buffer zone) in the middle of the screen
        buffer_zone_height = 10 * self.y_scale # Height of the buffer zone
        buffer_zone_width = self.screen.get_width()
        buffer_zone_rect = pygame.Rect(0, mid_y - buffer_zone_height // 2, buffer_zone_width, buffer_zone_height)

        # Add some transparency to the buffer zone by using a surface
        buffer_zone_surface = pygame.Surface((buffer_zone_width, buffer_zone_height))
        buffer_zone_surface.set_alpha(100)  # Set alpha for transparency
        buffer_zone_surface.fill((0, 0, 0))  # Fill with blue color (can change)

        # Blit the buffer zone surface to the screen
        self.screen.blit(buffer_zone_surface, (0, mid_y - buffer_zone_height // 2))
        for player in self.team1.players:
            pos = player.get_pos()
            grid_pos = self._grid_to_screen(pos)

            self.screen.blit(player.image, grid_pos)

        for player in self.team2.players:
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
        

    def _grid_to_screen(self, pos):
        # Convert grid position to screen coordinates
        screen_x = self.x_scale * (pos[0] - self.xmin)
        screen_y = 800 - self.y_scale * (pos[1] - self.ymin)  # Invert Y to match Pygame screen
        return (screen_x, screen_y)   

    @property
    # Return state
    def _state(self):
        pass
        

    def reset(self):
        self.team1.set_random_pos(self.team1_bounds[0], self.team1_bounds[1])
        self.team2.set_random_pos(self.team2_bounds[0], self.team2_bounds[1])
        return self._state