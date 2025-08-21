import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
from ur5_pick_place_env import UR5RobotiqEnv
from stable_baselines3.common.logger import configure


# Create logs and models directories
os.makedirs("./logs", exist_ok=True) 
os.makedirs("./models", exist_ok=True)

# Create and wrap the training environment
train_env = UR5RobotiqEnv(render_mode="No") # Initialize the environment with a render mode (human or No)
train_env = Monitor(train_env, "./logs/monitor.csv")

# Set up TensorBoard logging
logger = configure("./logs/tensorboard/", ["tensorboard"])

# Define SAC model
model = SAC( # Initialize SAC model with specified parameters
    "MlpPolicy", # Use Multi-Layer Perceptron policy
    train_env, # Use the wrapped training environment
    verbose=1,
    learning_rate=1e-4, # Set learning rate for the optimizer
    buffer_size=100000, # Set size of the replay buffer
    learning_starts=1000, # Number of steps before learning starts
    batch_size=512, # Batch size for training
    tau=0.005, # Soft update coefficient for target networks
    gamma=0.99, # Discount factor for future rewards
    train_freq=10, # Frequency of training (every 10 steps)
    gradient_steps=1, # Number of gradient steps per training
    ent_coef="auto", # Automatically adjust entropy coefficient
    tensorboard_log="./logs/tensorboard/", # Directory for TensorBoard logs
)
model.set_logger(logger)

# Define callbacks
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./models/", name_prefix="ur5_teacher_sac")

# Train the model
total_timesteps = 500000
model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback], log_interval=100, progress_bar=True)

# Save the final teacher model
model.save("./models/ur5_teacher_sac_final")

# Close environment
train_env.close()