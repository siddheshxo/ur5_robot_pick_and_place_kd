# Knowledge Distillation for Robotics Control Policies using Reinforcement Learning

## Thesis Overview
The thesis explores the application of Knowledge Distillation (KD) to enhance robotics control policies trained using reinforcement learning (RL). It utilizes the Soft Actor-Critic (SAC) algorithm to train a teacher model and student models (with and without KD) in a simulated UR5 robotic arm environment with a Robotiq 85 gripper. The study evaluates performance in a pick-and-place task, focusing on metrics such as Mean Reward, Episode Length, Success Rate, Actor and Critic Loss.

## Repository Structure
- *ur5_pick_place_env.py*: Custom Gym environment for the UR5 robotic arm pick-and-place task.
- *train_teacher.py*: Trains the Teacher model.
- *train_student_kd.py*: Trains the Student Model with Knowledge Distillation.
- *train_student_no_kd.py*: Trains the Student Model without Knowledge Distillation.
- *eval_teacher.py*: Evaluates the Teacher Model.
- *eval_student_kd.py*: Evaluates the Student Model with Knowledge Distillation.
- *eval_student_no_kd.py*: Evaluates the Student Model without Knowledge Distillation.
- *logs/*: Stores TensorBoard logs and evaluation metrics.
- *models/*: Stores trained model checkpoints and the complete model after completion.
- *meshes/*: Contains mesh files for the simulation.
- *urdf/*: Contains URDF files (e.g., plane.urdf, table/table.urdf, tray/tray.urdf, cube.urdf) for PyBullet.
- *plots/*: Directory containing plotting scripts:
  - *plotting_teacher/*, *plotting_student_kd/*, *plotting_student_no_kd/*, *plotting_eval_all_models/*: Directories created by plotting scripts to save generated plots.
  - *plot_teacher.py*: Plot the Teacher Model performance metrics from Tensorboard.
  - *plot_student_kd.py*: Plot the Student Model with Knowledge Distillation performance metrics from Tensorboard.
  - *plot_student_no_kd.py*: Plot the Student Model without Knowledge Distillation performance metrics from Tensorboard.
  - *plot_eval_reward_distribution_all_models.py*: Plot Evaluation performance metrics (Reward Distribution) of all the models.
  - *plot_eval_reward_and_success_rate_all_models.py*: Plot Evaluation performance metrics (Mean Reward and Success Rate) of all the models.
 
## Environment Details
- Action Space: Continuous control of end-effector target position in [x, y] position (z is automatically managed by waypoints).
- Observation Space: The [x, y] position of the target cube on the table.
- Reward Function:
  - Positive rewards for reaching the cube, successful grasp, lift, and placement in tray.
  - Bonus reward for faster completion (remaining steps).
  - Proximity-based reward for placing cube closer to tray center.
  - Large reward for full successful placement inside tray.
  - Negative rewards for boundary violations, collisions (table/tray), failed grasp, failed lift, improper release, or cube out of bounds.
- Episode Termination:
  - Task succeeds when the cube is placed in the tray.
  - Episode ends early if cube goes out of bounds.
  - Otherwise, episode ends when max steps are reached.

## Requirements
- Python 3.8+
- Required Libraries:
  - gymnasium
  - stable-baselines3
  - numpy
  - pybullet
  - pandas
  - torch
  - matplotlib
  - tensorboard
- Install dependencies using:

```bash
pip install gymnasium stable-baselines3 numpy pybullet pandas torch matplotlib tensorboard
```

## How to Run
**1. Clone the Repository:**

```bash
git clone https://github.com/siddheshxo/ur5_robot_pick_and_place_kd.git
cd ur5_robot_pick_and_place_kd
```

**2. Set Up the Environment:**
- Ensure URDF files (plane.urdf, table/table.urdf, tray/tray.urdf, cube.urdf) are in the *urdf/* directory or adjust ur5_pick_place_env.py to match their location.
- Ensure the *meshes/* directory contains any required mesh files for the simulation.

**3. Train Models:**
- Train the Teacher Model:

```bash
python train_teacher.py
```

- Train the Student Model without Knowledge Distillation:

```bash
python train_student_no_kd.py
```

- Train the Student Model with Knowledge Distillation:

```bash
python train_student_kd.py
```
Note: Ensure the teacher model (ur5_teacher_sac_final.zip) is in *models/* before training the KD student.

**4. Evaluate Models:**
- Evaluate the Teacher Model:

```bash
python eval_teacher.py
```

- Evaluate the Student Model without Knowledge Distillation:

```bash
python eval_student_no_kd.py
```

- Evaluate the Student Model with Knowledge Distillation:

```bash
python eval_student_kd.py
```

Results are saved in *logs/* as CSV files and logged to TensorBoard. 

Note: Ensure the corresponding trained model (e.g., ur5_teacher_sac_final.zip for eval_teacher.py) is present in *models/*.

**5. Visualize Training Metrics:**
- Plot Teacher Model metrics:

```bash
python plots/plot_teacher.py
```

- Plot Student Model without Knowledge Distillation metrics:

```bash
python plots/plot_student_no_kd.py
```

- Plot Student Model with Knowledge Distillation metrics:

```bash
python plots/plot_student_kd.py
```

Plots are saved in respective *plots/plotting_** directories.

Note: Ensure TensorBoard logs and CSV files in *logs/* are present for plotting


**6. Visualize Evaluation Metrics:**
- Plot reward distribution for all models:

```bash
python plots/plot_eval_reward_distribution_all_models.py
```

- Plot reward and success rate for all models:

```bash
python plots/plot_eval_reward_and_success_rate_all_models.py
```

Plots are saved in respective *plots/plotting_** directories.

Note: Ensure TensorBoard logs and CSV files in *logs/* are present for plotting

## Notes
- Set render_mode="human" in *ur5_pick_place_env.py* to visualize the simulation.
- Ensure sufficient disk space for logs, models, and plots.

## Reference
This project builds upon the baseline code from [leesweqq/ur5_reinforcement_learning_grasp_object](https://github.com/leesweqq/ur5_reinforcement_learning_grasp_object), a custom PyBullet environment for UR5 pick-and-place tasks trained using the SAC algorithm from Stable-Baselines3.
