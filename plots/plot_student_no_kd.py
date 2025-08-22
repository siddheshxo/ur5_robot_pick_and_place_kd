from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import os

# Paths
log_dir = "./../logs/tensorboard_student_no_kd/" # Path of Student without KD Tensorboard
plot_dir = "./plotting_student_no_kd/"
os.makedirs(plot_dir, exist_ok=True)

# Load TensorBoard logs
ea = event_accumulator.EventAccumulator(log_dir) # Initialize event accumulator with log directory
ea.Reload() # Load the TensorBoard event data

# Available tags and their proper plot titles
tag_title_map = { # Map of TensorBoard tags to human-readable plot titles
    'rollout/ep_len_mean': 'Mean Episode Length',
    'rollout/ep_rew_mean': 'Mean Episode Reward',
    'rollout/success_rate': 'Success Rate',
    'train/actor_loss': 'Actor Loss',
    'train/critic_loss': 'Critic Loss'
}

# Plotting parameters for figures
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (10,6)
})

# Loop over each tag and create plot
for tag, title in tag_title_map.items():
    if tag not in ea.Tags()["scalars"]: # Check if the tag exists in the scalar data
        print(f"Tag {tag} not found in logs, skipping.")
        continue

    events = ea.Scalars(tag) # Get all events for the current tag
    steps = [e.step for e in events] # Extract training steps from events
    values = [e.value for e in events] # Extract values from events

    plt.figure()
    plt.plot(steps, values, label=title, linewidth=2, color='tab:blue')
    plt.xlabel("Training Steps")
    plt.ylabel(title)
    plt.title(title, pad=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save figure
    filename = os.path.join(plot_dir, f"{tag.replace('/', '_')}.png") # Generate filename by replacing '/' with '_'
    plt.savefig(filename, dpi=300)
    plt.close()

print(f"All plots saved in {plot_dir}")