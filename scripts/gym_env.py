import gym
from gym import spaces
import numpy as np
from game import Game

class CTFEnv(gym.Env):
    """
    Gym wrapper for the Capture the Flag game.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, team_sprite_path, team_flag_path, T=200, screen_width=800, screen_height=800):
        super(CTFEnv, self).__init__()

        # Initialize the game
        self.game = Game(
            team_sprite_path,
            team_flag_path,
            T=T,
            screen_width=screen_width
            screen_height=screen_height
        )

        # Observation space
        obs_dim = 2 * len(self.game.team1.players) + 2 * 2 # Positions of players and flags (x, y)
        self.observation_space = spaces.Box(low=0, high=30, shape=(obs_dim,), dtype=np.float32)

        # Angles for team members
        self.action_space = spaces.Box(low=0, high=2 * np.pi, shape=(len(self.game.team1.players),), dtype=np.float32)

        self.opponent_policy = "stationary"
        self.opponent_model = None

    def set_opponent_policy(self, policy_type, model=None):
        self.opponent_policy = policy_type
        self.opponent_model = model

    def _get_opponent_action(self):
        """
        Generate actions for the opponent team based on the current policy.
        """
        if self.opponent_policy == "stationary":
            return np.zeros(len(self.game.team2.players))  # No movement
        elif self.opponent_policy == "random":
            return np.random.uniform(0, 2 * np.pi, len(self.game.team2.players))
        elif self.opponent_policy == "learned" and self.opponent_model:
            # Use the learned model to generate actions
            # Example: Pass team2's positions as input to the model
            team2_obs = self.game.team2.get_pos().flatten()
            return self.opponent_model(team2_obs)
        else:
            raise ValueError("Invalid opponent policy or missing model for 'learned' policy.")


    def reset(self):
        team1_pos, team2_pos = self.game.reset()

        obs = np.concatenate((team1_pos.flatten(), team2_pos.flatten()))
        return obs
    
    def step(self, action):
        team1_action = action
        team2_action = self._get_opponent_action()

        # Step the game
        done, t1_pos, t2_pos, t1_act, t2_act, team1_reward, team2_reward = self.game.step(
            team1_action, team2_action
        )

        # Flatten positions into a single array for observation
        obs = np.concatenate((t1_pos.flatten(), t2_pos.flatten()))

        reward = team1_reward - team2_reward

        return obs, reward, done, {}
    
    def render(self, mode='human'):
        self.game.render()

