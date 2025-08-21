import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import os
import time
import torch
from ur5_pick_place_env import UR5RobotiqEnv


# Create logs and models directories
os.makedirs("./logs", exist_ok=True)
os.makedirs("./models", exist_ok=True)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create and wrap the training environment
train_env = UR5RobotiqEnv(render_mode="No")
train_env = Monitor(train_env, "./logs/student_no_kd_monitor.csv")

# Set up TensorBoard logging
logger = configure("./logs/tensorboard_student_no_kd/", ["tensorboard"])

# Define a smaller SAC student model (same as KD version)
student_policy_kwargs = { 
    "net_arch": {
        "pi": [64, 64],  
        "qf": [128, 128],  
    }
}

# Initialize SAC model for the student with specified parameters similar to KD version
student_model = SAC(
    "MlpPolicy",
    train_env,
    policy_kwargs=student_policy_kwargs,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=10,
    gradient_steps=1,
    ent_coef="auto",
    tensorboard_log="./logs/tensorboard_student_no_kd/",
    device=device,
)
student_model.set_logger(logger)


total_timesteps = 100000
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./models/", name_prefix="ur5_student_no_kd")

start_time = time.time()
student_model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback], progress_bar=True)
training_time = time.time() - start_time
print(f"Training completed in {training_time / 60:.1f} minutes")

# Save final model
student_model.save("./models/ur5_student_no_kd_final")
train_env.close()