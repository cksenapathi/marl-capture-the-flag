from training import self_play_training, validation
from gym_env import CTFEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, ppo


def main():
    env = Monitor(CTFEnv(["../images/team1.png", "../images/team2.png"], ["../images/flag1.png", "../images/flag2.png"]), '../logs')
    self_play_training(env, save_dir="./models", total_timesteps=5000, self_play_epochs=10)

    opp_model = PPO.load("./models/team1/team1_550000_steps.zip")
    model = PPO.load("./models/team1/team1_540000_steps.zip")
    num_episodes = 1

    metrics = validation(env, num_episodes, model, opp_model)

if __name__ == "__main__":
    main()