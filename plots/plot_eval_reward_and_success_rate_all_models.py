import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
plot_dir = "./plotting_eval_all_models/"
os.makedirs(plot_dir, exist_ok=True)

# Load evaluation metrics files
teacher_metrics = pd.read_csv('./../logs/eval_metrics.csv')
kd_student_metrics = pd.read_csv('./../logs/student_eval_metrics.csv')
no_kd_student_metrics = pd.read_csv('./../logs/student_no_kd_eval_metrics.csv')

# Extract data
models = ['Teacher', 'KD Student', 'Non-KD Student'] # List of model names for labeling
mean_rewards = [teacher_metrics['mean_reward'][0], kd_student_metrics['mean_reward'][0], no_kd_student_metrics['mean_reward'][0]] # Extract rewards
success_rates = [teacher_metrics['success_rate'][0], kd_student_metrics['success_rate'][0], no_kd_student_metrics['success_rate'][0]] # Extract success rates

# Set up the plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (8, 5)

# Bar chart for Mean Reward
plt.figure(1)
plt.bar(models, mean_rewards, color=['blue', 'green', 'red'])
plt.title('Mean Reward by Model')
plt.ylabel('Mean Reward')
plt.ylim(0, max(mean_rewards) * 1.2)
for i, v in enumerate(mean_rewards): # Add text labels above each bar
    plt.text(i, v + 10, str(round(v, 2)), ha='center')

filename = os.path.join(plot_dir, f"mean_reward_barchart_all_models.png") # Generate filename for saving
plt.savefig(filename) # Save the figure
plt.close()

# Bar chart for Success Rate
plt.figure(2)
plt.bar(models, success_rates, color=['blue', 'green', 'red'])
plt.title('Success Rate by Model')
plt.ylabel('Success Rate (%)')
plt.ylim(0, 100)
for i, v in enumerate(success_rates): # Add text labels above each bar
    plt.text(i, v + 2, str(round(v, 2)), ha='center')

# Save figure
filename = os.path.join(plot_dir, f"success_rate_barchart_all_models.png") # Generate filename for saving
plt.savefig(filename)
plt.close()

print(f"All plots saved in {plot_dir}")