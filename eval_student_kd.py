import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import os
import pybullet as p
import time
import pandas as pd
from stable_baselines3.common.logger import configure
from ur5_pick_place_env import UR5RobotiqEnv

# Create logs directory
os.makedirs("./logs", exist_ok=True)

# Create evaluation environment
eval_env = UR5RobotiqEnv(render_mode="No")  # Set to "human" for visual debugging
eval_env = Monitor(eval_env, "./logs/student_eval_monitor.csv")

# Set up TensorBoard logger
logger = configure("./logs/tensorboard_student_eval/", ["tensorboard"])

# Load trained student model
model_path = "./models/ur5_student_sac_final.zip"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Student model file {model_path} not found. Ensure training was completed successfully.")

student_model = SAC.load(model_path, env=eval_env)
student_model.set_logger(logger)

# Evaluation parameters
n_eval_episodes = 100
log_path = "./logs"
eval_csv_path = os.path.join(log_path, "student_eval_metrics.csv")
eval_metrics = []

# Run evaluation
episode_rewards = []
successes = []

for episode in range(n_eval_episodes): # Iterate over the number of evaluation episodes
    obs, _ = eval_env.reset()
    done = False
    episode_reward = 0
    step_count = 0

    while not done: # Continue until episode is done
        action, _ = student_model.predict(obs, deterministic=True) # Predict action using the model with deterministic policy
        obs, reward, done, truncated, info = eval_env.step(action) # Take a step in the environment
        episode_reward += reward
        step_count += 1

        if eval_env.render_mode == "human":
            time.sleep(0.01)

        if done or truncated: # If episode is done or truncated
            success = info.get("is_success", False) # Get success status from info (default to False if not present)
            cube_pos = info.get("final_cube_pos", p.getBasePositionAndOrientation(eval_env.env.cube_id)[0])
            cube_vel = info.get("cube_velocity", p.getBaseVelocity(eval_env.env.cube_id)[0])

            print(f"Eval episode {episode + 1}: Final cube pos={cube_pos}, velocity={cube_vel}, success={success}")
            episode_rewards.append(episode_reward)
            successes.append(1 if success else 0)
            break

# Compute metrics
mean_reward = np.mean(episode_rewards)
success_rate = np.mean(successes) * 100
print(f"\nStudent Model Evaluation over {n_eval_episodes} episodes:")
print(f"Mean reward = {mean_reward:.2f}")
print(f"Success rate = {success_rate:.2f}%")

# Log to TensorBoard
logger.record("eval/mean_reward", mean_reward)
logger.record("eval/success_rate", success_rate)

# Save to CSV
eval_metrics.append({
    "timestep": student_model.num_timesteps,
    "mean_reward": mean_reward,
    "success_rate": success_rate
})
pd.DataFrame(eval_metrics).to_csv(eval_csv_path, index=False)

# Close environment
eval_env.close()