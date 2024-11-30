import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game import Game
from typing import Optional
import time

class CTFEnv(gym.Env):
    """
    Gym wrapper for the Capture the Flag game.
    """

    metadata = {"render.modes": ["human"], "render_fps": 30}

    def __init__(self, team_sprite_path, team_flag_path, T=200, screen_width=800, screen_height=800):
        super().__init__()

        # Initialize the game
        self.game = Game(
            team_sprite_path,
            team_flag_path,
            T=T,
            screen_width=screen_width,
            screen_height=screen_height
        )

        # Observation space
        # [team1 players x and y, team1 flag x and y, team2 players x and y, team2 flag x and y]
        obs_dim = 2 * (2 * len(self.game.team1.players) + 2) # Positions of players and flags (x, y)
        self.observation_space = spaces.Box(low=0, high=30, shape=(obs_dim,), dtype=np.float32)

        # Angles for team members (scaled)
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.game.team1.players),), dtype=np.float32)

        # self.action_space = spaces.Box(low=0, high=2 * np.pi, shape=(len(self.game.team1.players),), dtype=np.float32)

        self.opponent_policy = "stationary"
        self.opponent_model = None

    def rescale_action(self, action):
        # Rescale action from [-1, 1] to [0, 2π]
        if action is not None:
            return (action + 1) * np.pi
        else:
            return None

    def set_opponent_policy(self, policy_type, model=None):
        self.opponent_policy = policy_type
        self.opponent_model = model

    def _get_opponent_action(self):
        """
        Generate actions for the opponent team based on the current policy.
        """
        if self.opponent_policy == "stationary":
            return None  # No movement
        elif self.opponent_policy == "random":
            return np.random.uniform(0, 2 * np.pi, len(self.game.team2.players))
        elif self.opponent_policy == "learned" and self.opponent_model:
            # Current state information
            t1_pos = self.game.team1.get_pos().flatten()  # Team 1 player positions
            t2_pos = self.game.team2.get_pos().flatten()  # Team 2 player positions

            # Flip positions to make it look like Team 2's side is the "home side"
            board_width = self.game.board_width
            board_height = self.game.board_height
            
            # Flip x-coordinates and y-coordinates for both teams and flags
            flipped_t1_pos = np.array([[board_width - x, board_height - y] for x, y in t1_pos.reshape(-1, 2)]).flatten()
            flipped_t2_pos = np.array([[board_width - x, board_height - y] for x, y in t2_pos.reshape(-1, 2)]).flatten()

            # Transform the observation to Team 2's perspective
            obs_from_team2_perspective = np.concatenate((flipped_t2_pos, flipped_t1_pos))
            
            # Use the opponent model to predict actions
            if hasattr(self.opponent_model, "predict"):
                opponent_action = self.opponent_model.predict(obs_from_team2_perspective)[0]  # Assuming the model's predict method outputs actions
            else:
                raise ValueError("Provided opponent model does not have a 'predict' method.")

            flipped_opponent_action = np.array([(action + np.pi) % (2 * np.pi) for action in opponent_action])

            # Return the adjusted action for Team 1 (same action space dimension)
            return flipped_opponent_action


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        team1_pos, team2_pos = self.game.reset()

        obs = np.concatenate((team1_pos.flatten(), team2_pos.flatten()))
        return np.array(obs, dtype=np.float32), {}
    
    def step(self, action):
        team1_action = self.rescale_action(action)
        team2_action = self.rescale_action(self._get_opponent_action())

        # Step the game
        done, t1_pos, t2_pos, t1_act, t2_act, team1_reward, team2_reward = self.game.step(
            team1_action, team2_action
        )

        flag1 = self.game.team1.flag_pos
        flag2 = self.game.team2.flag_pos

        # Flatten positions into a single array for observation
        obs = np.concatenate((t1_pos.flatten(), flag1.flatten(), t2_pos.flatten(), flag2.flatten()))

        reward = team1_reward
        truncated = False

        if done:
            print("Game done.")
        return np.array(obs, dtype=np.float32), reward, bool(done), truncated, {}
    
    def render(self, mode='human'):
        self.game.render()

    def close(self):
        pass

