from training import self_play_training
from gym_env import CTFEnv
from stable_baselines3.common.env_checker import check_env


def main():
    env = CTFEnv(["../images/team1.png", "../images/team2.png"], ["../images/flag1.png", "../images/flag2.png"])
    self_play_training(env, save_dir="./models", total_timesteps=5000, self_play_epochs=10)

if __name__ == "__main__":
    main()