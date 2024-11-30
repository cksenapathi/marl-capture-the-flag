import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

def train_agent(env, agent_name, save_dir, total_timesteps, epoch, model=None, render_interval=100):
    """
    Tran a PPO agent
    """
    if model is None:
        model = PPO("MlpPolicy", env, verbose=1)

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=save_dir, name_prefix=agent_name)

    # Train the model
    # Train the model
    for timestep in range(total_timesteps):
        model.learn(total_timesteps=1, reset_num_timesteps=False, callback=checkpoint_callback)
        
        # Render the environment every `render_interval` timesteps
        if timestep % render_interval == 0:
            env.render()

    save_path = os.path.join(save_dir, f"{agent_name}_epoch_{epoch}.zip")
    model.save(save_path)
    print(f"{agent_name} policy saved to {save_path}")

    return model

def self_play_training(env, save_dir, total_timesteps, self_play_epochs):
    """
    Perform self-play trianing for two agents
    """

    team1_dir = os.path.join(save_dir, "team1")
    team2_dir = os.path.join(save_dir, "team2")

    # Ensure dirs exist
    os.makedirs(team1_dir, exist_ok=True)
    os.makedirs(team2_dir, exist_ok=True)

    model_team1 = None

    for epoch in range(self_play_epochs):
        print(f"=== Epoch {epoch + 1}/{self_play_epochs} ===")

        if epoch == 0:
            print("Using stationary Team 2 for initial training")
            env.set_opponent_policy("stationary", None)
        else:
            print("Setting team 2's policy to previously trained team 1 policy")
            prev_policy = os.path.join(team1_dir, f"team1_epoch_{epoch - 1}.zip")

            if os.path.exists(prev_policy):
                env.set_opponent_policy("learned", PPO.load(prev_policy))
            else:
                print(f"Warning: Previous policy {prev_policy} not found. Using stationary opponent instead.")
                env.set_opponent_policy("stationary", None)

        # Train Team 1
        print("Training Team 1...")
        model_team1 = train_agent(env, "team1", team1_dir, total_timesteps, epoch, model=model_team1)    


    print("Self play training complete")