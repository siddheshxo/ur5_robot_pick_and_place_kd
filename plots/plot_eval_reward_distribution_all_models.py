import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Paths
plot_dir = "./plotting_eval_all_models/"
os.makedirs(plot_dir, exist_ok=True)

# Load monitor files
teacher_monitor = pd.read_csv('./../logs/eval_monitor.csv', skiprows=1) # Load teacher monitor data, skipping header row (contains garbage characters)
kd_student_monitor = pd.read_csv('./../logs/student_eval_monitor.csv', skiprows=1) # Load KD student monitor data, skipping header row (contains garbage characters)
no_kd_student_monitor = pd.read_csv('./../logs/student_no_kd_eval_monitor.csv', skiprows=1) # Load non-KD student monitor data, skipping header row (contains garbage characters)

# Extract rewards
teacher_rewards = teacher_monitor['r']
kd_student_rewards = kd_student_monitor['r']
no_kd_student_rewards = no_kd_student_monitor['r']

# Create episode indices (1 to 100)
episodes = range(1, 101)

# Compute moving average (window size = 5) for smoothing trends
def moving_average(data, window=5):
    # Define function to calculate moving average
    return np.convolve(data, np.ones(window)/window, mode='same')  # Apply convolution with uniform weights, matching input length

ma_window = 5
teacher_ma = moving_average(teacher_rewards, ma_window) # Compute moving average for teacher rewards
kd_student_ma = moving_average(kd_student_rewards, ma_window) # Compute moving average for KD student rewards
no_kd_student_ma = moving_average(no_kd_student_rewards, ma_window) # Compute moving average for non-KD student rewards

# Legend customization parameters
legend_loc = 'center'
legend_bbox = (0.5, 1.15)
legend_fontsize = 8 

# Set up plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 12) 

# Create stacked subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)

# Plot for Teacher
ax1.scatter(episodes, teacher_rewards, color='blue', label='Teacher Rewards', alpha=0.5, s=30)
ax1.plot(episodes, teacher_ma, color='navy', label=f'Teacher Moving Avg (window={ma_window})', linewidth=2)
ax1.set_title('Teacher Reward per Episode')
ax1.set_ylabel('Reward')
ax1.legend(loc=legend_loc, bbox_to_anchor=legend_bbox, fontsize=legend_fontsize, frameon=True)
ax1.grid(True)

# Plot for Student with KD
ax2.scatter(episodes, kd_student_rewards, color='green', label='Student with KD Rewards', alpha=0.5, s=30)
ax2.plot(episodes, kd_student_ma, color='darkgreen', label=f'Student with KD Moving Avg (window={ma_window})', linewidth=2)
ax2.set_title('Student with KD Reward per Episode')
ax2.set_ylabel('Reward')
ax2.legend(loc=legend_loc, bbox_to_anchor=legend_bbox, fontsize=legend_fontsize, frameon=True)
ax2.grid(True)

# Plot for Student without KD
ax3.scatter(episodes, no_kd_student_rewards, color='red', label='Student without KD Rewards', alpha=0.5, s=30)
ax3.plot(episodes, no_kd_student_ma, color='darkred', label=f'Student without KD Moving Avg (window={ma_window})', linewidth=2)
ax3.set_title('Student without KD Reward per Episode')
ax3.set_xlabel('Episode Number')
ax3.set_ylabel('Reward')
ax3.legend(loc=legend_loc, bbox_to_anchor=legend_bbox, fontsize=legend_fontsize, frameon=True)
ax3.grid(True)

# Adjust layout and save
plt.tight_layout()

# Save figure
filename = os.path.join(plot_dir, f"per_episode_reward_distribution_all_models.png") # Generate filename for saving
plt.savefig(filename)
plt.close()

print(f"All plots saved in {plot_dir}")