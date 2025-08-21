import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import torch
import torch.nn as nn
import os
from ur5_pick_place_env import UR5RobotiqEnv
import time

# Create logs directory
os.makedirs("./logs", exist_ok=True)
os.makedirs("./models", exist_ok=True)

# Set device to CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

teacher_model_path = "./models/ur5_teacher_sac_final.zip" # Define path to the teacher model
if not os.path.exists(teacher_model_path):
    raise FileNotFoundError(f"Teacher model file {teacher_model_path} not found.")

teacher_model = SAC.load(teacher_model_path) # Load the teacher SAC model
teacher_model.policy.to(device) # Move teacher policy to the selected device

train_env = UR5RobotiqEnv(render_mode="No")
train_env = Monitor(train_env, "./logs/student_monitor.csv")

logger = configure("./logs/tensorboard_student/", ["tensorboard"])

student_policy_kwargs = { # Define policy network architecture for the student
    "net_arch": {
        "pi": [64, 64], # Architecture for the policy network (actor)
        "qf": [128, 128], # Architecture for the Q-function network (critic)
    }
}
student_model = SAC(
    "MlpPolicy",
    train_env,
    policy_kwargs=student_policy_kwargs, # Pass the custom policy architecture
    verbose=1,
    learning_rate=1e-4,
    buffer_size=50000, # Set size of the replay buffer
    learning_starts=1000,
    batch_size=256, # Batch size for training
    tau=0.005,
    gamma=0.99,
    train_freq=10,
    gradient_steps=1,
    ent_coef="auto",
    tensorboard_log="./logs/tensorboard_student/",
    device=device, # Set device for training
)
student_model.set_logger(logger)


def compute_actor_kd_loss(student_actions, teacher_actions, temperature=2.0):
    # Define loss function for actor knowledge distillation
    teacher_soft = teacher_actions / temperature # Apply temperature scaling to teacher actions
    student_soft = student_actions / temperature # Apply temperature scaling to student actions
    return nn.MSELoss()(student_soft, teacher_soft) # Return mean squared error between softened actions

def compute_critic_kd_loss(student_q, teacher_q, temperature=2.0):
    # Define loss function for critic knowledge distillation
    scale = temperature * 1000.0  # prevent large gradients
    return nn.MSELoss()(student_q / scale, teacher_q / scale) # Return mean squared error between scaled Q-values


class KDLossCallback(BaseCallback): # Define custom callback class for knowledge distillation
    def __init__(self, teacher_model, kd_weight=0.9, temperature=2.0, warmup_steps=15000, verbose=0):
        super().__init__(verbose) # Call parent constructor with verbosity
        self.teacher_model = teacher_model
        self.base_kd_weight = kd_weight
        self.base_temperature = temperature
        self.warmup_steps = warmup_steps
        self.episode_count = 0

    def inject_teacher_success(self):
        # Run teacher for a short episode and insert transitions into replay buffer
        reset_result = self.training_env.reset() # Reset the training environment
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        done = False
        while not done: # Continue until episode is done
            action, _ = self.teacher_model.predict(obs, deterministic=False) # Predict action from teacher
            
            step_result = self.training_env.step(action) # Take a step with teacher action
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                next_obs, reward, done, info = step_result
            else:
                raise RuntimeError(f"Unexpected step() output length: {len(step_result)}")
            

            if reward >= 300:
                self.model.replay_buffer.add( # Add transition to replay buffer
                    np.array([obs]),
                    np.array([next_obs]),
                    np.array([action]),
                    np.array([reward]),
                    np.array([done]),
                    [{}]
                )
            obs = next_obs

    def _on_step(self) -> bool: # Method called at each step
        total_timesteps = self.model._total_timesteps # Get total timesteps

        # Warmup phase - no KD
        if self.num_timesteps < self.warmup_steps:
            return True # Continue without KD

        # Periodically inject teacher-success data
        if self.num_timesteps % 2000 == 0: # Every 2000 steps
            self.inject_teacher_success() # Inject teacher-generated data

        # Only run KD if enough data in buffer
        if self.model.replay_buffer.size() < self.model.batch_size: # If buffer size is insufficient
            return True # Continue without KD

        # Anneal KD weight
        progress = self.num_timesteps / total_timesteps # Calculate training progress
        kd_max, kd_min = 0.90, 0.70 # Maximum and minimum KD weights
        hold = 0.40  # Hold high KD for 40% of training
        if progress <= hold:
            kd_weight = kd_max
        else:
            t = (progress - hold) / (1.0 - hold)       # rescale to [0,1]
            kd_weight = kd_min + 0.5*(kd_max - kd_min)*(1 + np.cos(np.pi * (1 - t)))
            kd_weight = max(kd_weight, kd_min)  # Ensure weight doesn't go below minimum

        # Anneal temperature
        if self.num_timesteps < 0.2 * total_timesteps: # First 20% of training
            temperature = 5.0
        else:
            temperature = self.base_temperature

        # Sample from replay buffer
        replay_data = self.model.replay_buffer.sample(self.model.batch_size, env=self.model._vec_normalize_env)

        obs_tensor = replay_data.observations.to(device).float() # Convert observations to tensor and move to device

        # Teacher actions
        obs_np = replay_data.observations.cpu().numpy() # Convert observations to NumPy on CPU
        with torch.no_grad():
            teacher_actions_np, _ = self.teacher_model.predict(obs_np, deterministic=False) # Predict teacher actions
        teacher_actions = torch.tensor(teacher_actions_np, dtype=torch.float32, device=device) # Convert to tensor on device

        # Student actions
        student_actions = self.model.actor(obs_tensor, deterministic=False)  # Predict student actions
        student_actions = student_actions.to(device).float()  # Move to device and ensure float type
        if student_actions.dim() == 1: 
            student_actions = student_actions.unsqueeze(0) # Add batch dimension

        # Match shape
        if teacher_actions.shape != student_actions.shape: # If shapes don't match
            if student_actions.dim() == 1: # If student actions are 1D
                student_actions = student_actions.unsqueeze(0).repeat(teacher_actions.size(0), 1) # Repeat to match teacher shape
            else:
                raise RuntimeError(f"Shape mismatch: teacher {teacher_actions.shape}, student {student_actions.shape}") # Raise error for mismatch

        # Actor KD loss
        actor_kd_loss = compute_actor_kd_loss(student_actions, teacher_actions, temperature)

        # Teacher Q-values
        with torch.no_grad(): # Disable gradient computation
            teacher_q_values = self.teacher_model.critic.q_networks[0](torch.cat([obs_tensor, teacher_actions], dim=-1))

        # Student Q-values
        student_q_values = self.model.critic.q_networks[0](torch.cat([obs_tensor, student_actions], dim=-1))

        # Critic KD loss
        critic_kd_loss = compute_critic_kd_loss(student_q_values, teacher_q_values, temperature)

        # Dynamic KD scaling to match RL loss scale
        with torch.no_grad():
            actor_loss_val = abs(actor_kd_loss.item()) + 1e-8 # Get actor loss value with small epsilon
        kd_scale = actor_loss_val / (actor_kd_loss.item() + 1e-8) # Compute scaling factor

        total_kd_loss = kd_weight * actor_kd_loss + (1 - kd_weight) * critic_kd_loss # Combine losses with weight
        total_kd_loss = total_kd_loss * kd_scale # Apply scaling

        # Update actor
        self.model.actor.optimizer.zero_grad() # Clear previous gradients
        total_kd_loss.backward() # Backpropagate KD loss
        self.model.actor.optimizer.step() # Update actor parameters

        # Log in Tensorboard
        self.logger.record("train/actor_kd_loss", actor_kd_loss.item())
        self.logger.record("train/critic_kd_loss", critic_kd_loss.item())
        self.logger.record("train/total_kd_loss", total_kd_loss.item())
        self.logger.record("train/kd_weight", kd_weight)
        self.logger.record("train/kd_temperature", temperature)

        if self.locals.get("infos") and len(self.locals["infos"]) > 0:
            info = self.locals["infos"][-1]
            if "episode" in info:
                self.episode_count += 1
                print(f"[Episode {self.episode_count}] ActorKD={actor_kd_loss.item():.4f}, CriticKD={critic_kd_loss.item():.4f}, KDweight={kd_weight:.3f}, Temp={temperature}")

        return True


total_timesteps = 100000
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./models/", name_prefix="ur5_student_sac")
kd_callback = KDLossCallback(teacher_model, kd_weight=0.9, temperature=2.0, warmup_steps=15000) # Create KD callback with specified parameters

start_time = time.time()
student_model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, kd_callback], progress_bar=True) # Train the student model with callbacks
training_time = time.time() - start_time
print(f"Training completed in {training_time / 60:.1f} minutes")

student_model.save("./models/ur5_student_sac_final")
train_env.close()