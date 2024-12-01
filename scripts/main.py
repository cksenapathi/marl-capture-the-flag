from training import self_play_training
from gym_env import CTFEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, ppo


def main():
    env = CTFEnv(["../images/team1.png", "../images/team2.png"], ["../images/flag1.png", "../images/flag2.png"])
    # self_play_training(env, save_dir="./models", total_timesteps=5000, self_play_epochs=10)

    opp_model = PPO.load("./models/team1/team1_550000_steps.zip")
    env.set_opponent_policy("learned", opp_model)
    model = PPO.load("./models/team1/team1_540000_steps.zip")

    obs, info = env.reset()

    done = False
    while not done:
        act = model.predict(obs)[0]
        obs, reward, done, trunc, info = env.step(act)
        env.render()

if __name__ == "__main__":
    main()