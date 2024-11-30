import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

def train_agent(env, agent_name, save_dir, total_timestaps):
    """
    Tran a PPO agent
    """

    os.makedris(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{agent_name}.zip")

    # Load existing model or initialize new one
    if os.path.exists(model_path):
        print(f"Loading model for {agent_name}")
        model = PPO.load(model_path, env)
    else:
        print(f"Creating new model for {agent_name}")
        model = PPO("MlpPolicy", env, verbose=1)

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=save_dir, name_prefix=agent_name)

    # Train the model
    model.learn(total_timesteps=total_timestaps, callback=checkpoint_callback)

    model.save(model_path)
    print(f"Model saved for {agent_name} at {model_path}")

def self_play_training(env, save_dir, total_timesteps, self_play_epochs):
    """
    Perform self-play trianing for two agents
    """

    team1_dir = os.path.join(save_dir, "team1")
    team2_dir = os.path.join(save_dir, "team2")

    for epoch in range(self_play_epochs):
        print(f"=== Epoch {epoch + 1}/{self_play_epochs} ===")

        print("Training team 1...")
        env.game.team2.set_policy(os.path.join(team2_dir, "team2.zip"))
        train_agent(env, "team1", team1_dir, total_timesteps)

        # Train team 2 against team 2
        print("Training team 2...")
        env.game.team1.set_policy(os.path.join(team1_dir, "team1.zip"))
        train_agent(env, "team2", team2_dir, total_timesteps)

    print("Self play training complete")