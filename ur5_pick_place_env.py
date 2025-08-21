import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
from collections import namedtuple
import math

class UR5RobotiqEnv(gym.Env):
    def __init__(self, render_mode="No"): #Cite: https://github.com/leesweqq/ur5_reinforcement_learning_grasp_object/blob/main/ur5_env.py and also modified
        # Initialize the environment with a render mode (human or No)
        super(UR5RobotiqEnv, self).__init__()
        self.render_mode = render_mode
        self.episode_count = 0 

        # Connect to PyBullet with GUI if render_mode is "human", otherwise use DIRECT mode
        connection_mode = p.GUI if render_mode == "human" else p.DIRECT
        self.physics_client = p.connect(connection_mode)
        p.setGravity(0, 0, -9.8) # Set gravity
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # Set path for PyBullet data
        p.setTimeStep(1 / 300) # Set simulation time step

        # Define action space: [x, y] target position for the end-effector
        self.action_space = spaces.Box(low=np.array([0.3, -0.3]), high=np.array([0.7, 0.3]), dtype=np.float64)

        # Define observation space: [x, y] position of the target cube
        self.observation_space = spaces.Box(low=np.array([0.3, -0.3]), high=np.array([0.7, 0.3]), dtype=np.float64)

        # Load environment objects
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
        self.tray_id = p.loadURDF("tray/tray.urdf", [0.5, 0.9, 0.3], p.getQuaternionFromEuler([0, 0, 0]))
        self.cube_id2 = p.loadURDF("cube.urdf", [0.5, 0.9, 0.3], p.getQuaternionFromEuler([0, 0, 0]), globalScaling=0.6, useFixedBase=True)

        # oad the robot (UR5 with Robotiq 85 gripper)
        self.robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
        self.robot.load()

        # Enable collision between robot links (except base) and table/tray
        for link_idx in range(1, p.getNumJoints(self.robot.id)):
            p.setCollisionFilterPair(self.robot.id, self.table_id, link_idx, -1, 1)
            p.setCollisionFilterPair(self.robot.id, self.tray_id, link_idx, -1, 1)

        # Initialize cube and environment boundaries
        self.cube_id = None
        self.max_steps = 200 # Maximum steps per episode
        self.current_step = 0 # Current step counter
        self.tray_pos = np.array([0.5, 0.9]) # Tray position
        self.table_height = 0.63 # Table height
        self.boundary_x = [0.3, 0.7] # X-axis boundaries
        self.boundary_y = [-0.3, 0.3] # Y-axis boundaries
        self.tray_bounds_x = [0.23, 0.77]  # Relaxed X bounds for tray
        self.tray_bounds_y = [0.64, 1.12] # Relaxed Y bounds for tray
        self.tray_center_bounds_x = [0.42, 0.58]  # Tighter X bounds for tray center
        self.tray_center_bounds_y = [0.82, 0.98] # Tighter Y bounds for tray center
        self.tray_max_z = 0.65 # Maximum Z height for tray

        if self.render_mode == "human":
            self.draw_boundary(self.boundary_x, self.boundary_y, self.table_height)

    def draw_boundary(self, x_range, y_range, z_height): # Implemented from https://github.com/leesweqq/ur5_reinforcement_learning_grasp_object/blob/main/ur5_env.py
        # Draw boundary lines in the environment for visualization
        corners = [
            [x_range[0], y_range[0], z_height],
            [x_range[1], y_range[0], z_height],
            [x_range[1], y_range[1], z_height],
            [x_range[0], y_range[1], z_height],
        ]
        for i in range(len(corners)):
            p.addUserDebugLine(corners[i], corners[(i + 1) % len(corners)], [1, 0, 0], lineWidth=2)

    def reset(self, seed=None, options=None): # Complete function implementation from https://github.com/leesweqq/ur5_reinforcement_learning_grasp_object/blob/main/ur5_env.py and modified.
        # Reset the environment to initial state
        self.current_step = 0
        self.episode_count += 1
        self.robot.original_position(self.robot, render_mode=self.render_mode) # Reset robot to original position

        # Debug: Print all joint names to verify gripper_links
        #print("Available joint names:")
        #for i in range(p.getNumJoints(self.robot.id)):
        #    print(p.getJointInfo(self.robot.id, i)[1].decode("utf-8"))

        x_range = np.arange(0.45, 0.65, 0.1) # Define range for x-coordinate
        y_range = np.arange(-0.2, 0.2, 0.1) # Define range for y-coordinate
        cube_start_pos = [ # Randomly select initial cube position
            np.random.choice(x_range),
            np.random.choice(y_range),
            self.table_height
        ]
        cube_start_orn = p.getQuaternionFromEuler([0, 0, 0])

        if self.cube_id is not None: # If cube already exists, reset its position
            p.resetBasePositionAndOrientation(self.cube_id, cube_start_pos, cube_start_orn)
            p.changeDynamics(self.cube_id, -1, lateralFriction=10.0) 
        else: # If cube doesn't exist, load it
            self.cube_id = p.loadURDF("./urdf/cube_blue.urdf", cube_start_pos, cube_start_orn)
            p.changeDynamics(self.cube_id, -1, lateralFriction=10.0)

        for _ in range(50): # Simulate for 50 steps to stabilize
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(0.01)

        self.initial_cube_pos = np.array(cube_start_pos[:2]) # Store initial x, y position
        self.target_pos = np.array(cube_start_pos[:2]) # Set target position
        observation = self.target_pos # Return target position as observation
        info = {"terminal_observation": observation}
        print(f"Reset: Cube at {cube_start_pos[:2]}")
        return observation, info
    
    def check_gripper_cube_contact(self):
        # Check if the gripper is in contact with the cube
        gripper_links = ['left_inner_finger_pad', 'right_inner_finger_pad', 'left_inner_finger', 'right_inner_finger']
        contact_detected = False # Initialize contact flag
        for link_idx in range(p.getNumJoints(self.robot.id)): # Iterate over all joints
            link_name = p.getJointInfo(self.robot.id, link_idx)[1].decode("utf-8")
            if link_name in gripper_links: # Check if joint is a gripper link
                contacts = p.getContactPoints(self.robot.id, self.cube_id, link_idx, -1)
                if contacts: # If contact is detected
                    contact_detected = True # Set flag to true
                    break
        return contact_detected

    def step(self, action): # Implemented from https://github.com/leesweqq/ur5_reinforcement_learning_grasp_object/blob/main/ur5_env.py and modified
        # Execute one step in the environment based on the given action
        self.current_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        eef_state = p.getLinkState(self.robot.id, self.robot.eef_id) # Get end-effector state
        eef_orientation = eef_state[1] # Extract orientation

        # Waypoint 1: Move above cube (z=0.88)
        target_pos = np.array([action[0], action[1], 0.88]) # Set target position above cube
        self.robot.move_arm_ik(target_pos, eef_orientation) # Move arm using inverse kinematics
        for _ in range(100):
            p.stepSimulation()

        eef_state = self.robot.get_current_ee_position() # Get updated end-effector position
        eef_position = np.array(eef_state[0])[:2] # Extract x, y coordinates
        distance_to_target = np.linalg.norm(eef_position - self.target_pos) # Calculate distance to target

        boundary_penalty = -50 if (eef_position[0] < self.boundary_x[0] or eef_position[0] > self.boundary_x[1] or
                                  eef_position[1] < self.boundary_y[0] or eef_position[1] > self.boundary_y[1]) else 0 # Apply penalty for boundary violation

        cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube_id)
        cube_xy = np.array(cube_pos[:2])
        if cube_pos[2] < self.table_height - 0.3 or (cube_xy[0] < self.boundary_x[0] or cube_xy[0] > self.boundary_x[1] or
                                                     cube_xy[1] < self.boundary_y[0] or cube_xy[1] > self.boundary_y[1]): # Check if cube is out of bounds
            reward = -10 + boundary_penalty # Apply penalty
            print(f"Reset: Cube out of bounds at {cube_pos}, reward: {reward}")
            observation, info = self.reset()
            info["terminal_observation"] = observation
            return observation, reward, True, False, info

        reward = -5 * distance_to_target + boundary_penalty # Calculate reward based on distance and boundary
        done = False
        truncated = False
        info = {}

        if distance_to_target <= 0.02: # Check if close to target
            steps_taken = self.max_steps - self.current_step
            reward = 100 + max(0, steps_taken * 1) + boundary_penalty # Update reward
            print(f"Attempting pick at {self.target_pos[0], self.target_pos[1]}, distance: {distance_to_target:.4f}, reward: {reward}")

            # Waypoint 2: Pre-grasp alignment (z=0.8)
            target_pos = np.array([action[0], action[1], 0.8]) # Set pre-grasp position
            self.robot.move_arm_ik(target_pos, eef_orientation) # Move arm to pre-grasp position
            for _ in range(100):
                p.stepSimulation()
                if self.render_mode == "human":
                    time.sleep(0.01)
                    p.addUserDebugLine([action[0], action[1], 0.8], [action[0], action[1], 0.8 + 0.1], [0, 1, 0], lifeTime=0.1)
                    p.addUserDebugLine([action[0], action[1], 0.74], [action[0], action[1], 0.74 + 0.1], [1, 0, 0], lifeTime=0.1)

            # Pre-grasp pause for stability
            for _ in range(100):
                p.stepSimulation()
                if self.render_mode == "human":
                    time.sleep(0.01)

            # Check for arm-table collision
            table_contacts = []
            link_names = {}
            for link_idx in range(1, p.getNumJoints(self.robot.id)): # Iterate over links except base
                contacts = p.getContactPoints(self.robot.id, self.table_id, link_idx, -1) # Get contacts with table
                if contacts:
                    link_name = p.getJointInfo(self.robot.id, link_idx)[1].decode("utf-8")
                    link_names[link_idx] = link_name
                    table_contacts.extend(contacts)
            if table_contacts: # If collision detected
                reward -= 50 # Apply penalty
                contact_positions = [cp[5] for cp in table_contacts]
                colliding_links = [link_names.get(cp[3], "unknown") for cp in table_contacts]
                print(f"Collision with table detected at pre-grasp, positions: {contact_positions}, links: {colliding_links}, reward: {reward}")
                if self.render_mode == "human":
                    p.addUserDebugText("Table Collision", [0.5, 0, 0.9], textColorRGB=[1, 0, 0], textSize=2, lifeTime=0.5)
                observation, info = self.reset()
                info["terminal_observation"] = observation
                return observation, reward, True, False, info

            # Log gripper link positions
            gripper_links = ['robotiq_85_base_link', 'finger_joint', 'left_inner_finger', 'right_inner_finger', 'left_inner_finger_pad', 'right_inner_finger_pad']
            gripper_positions = {}
            for link_idx in range(p.getNumJoints(self.robot.id)):
                link_name = p.getJointInfo(self.robot.id, link_idx)[1].decode("utf-8")
                if link_name in gripper_links:
                    link_state = p.getLinkState(self.robot.id, link_idx)
                    gripper_positions[link_name] = link_state[0]
            if not gripper_positions:
                eef_state = self.robot.get_current_ee_position()
                gripper_positions['end_effector'] = eef_state[0]
            print(f"Pre-grasp gripper positions: {gripper_positions}")

            # Two-stage gripper closing
            self.robot.move_gripper(0.035, max_force=100000) # First stage: partial close
            for _ in range(150):
                p.stepSimulation()
                if self.render_mode == "human":
                    time.sleep(0.01)
            self.robot.move_gripper(0.0002, max_force=100000) # Second stage: full close
            for _ in range(150):
                p.stepSimulation()
                if self.render_mode == "human":
                    time.sleep(0.01)

            # Pause to stabilize grasp
            for _ in range(50):
                p.stepSimulation()
                if self.render_mode == "human":
                    time.sleep(0.01)

            # Check gripper contact with cube
            if self.check_gripper_cube_contact(): # If contact is detected
                reward += 50 # Add reward for successful grasp
                print(f"Grasp success: Contact detected, reward: {reward}")
            else:
                reward -= 20 # Apply penalty
                print(f"Grasp failure: No contact detected, reward: {reward}")

            # Waypoint 3: Lift cube (z=1.0)
            target_pos = np.array([action[0], action[1], 1.0]) # Set lift position
            self.robot.move_arm_ik(target_pos, eef_orientation, max_velocity=3) # Move arm with reduced velocity
            for _ in range(100):
                p.stepSimulation()
                if self.render_mode == "human":
                    time.sleep(0.01)
                    p.addUserDebugLine([action[0], action[1], 1.0], [action[0], action[1], 1.0 + 0.1], [0, 0, 1], lifeTime=0.1)

            # Check if cube is lifted
            cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube_id) # Get cube position and orientation
            if cube_pos[2] <= self.table_height + 0.1: # Cube lift check
                reward -= 50 # Add reward for lifting
                print(f"Failed to lift cube, z: {cube_pos[2]:.4f}, reward: {reward}")
                observation, info = self.reset()
                info["terminal_observation"] = observation
                return observation, reward, True, False, info
            else: # If not lifted
                reward += 50 # Apply penalty
                print(f"Cube lifted successfully, z: {cube_pos[2]:.4f}, reward: {reward}")
                if self.render_mode == "human":
                    p.addUserDebugText("Success Pick", textColorRGB=[0, 0, 255], textPosition=[0.5, -0.3, 0.9], textSize=2, lifeTime=0.5)
                    time.sleep(0.5)

            # Move above tray (z=0.95, randomized x,y within center bounds)
            approach_x = np.mean(self.tray_center_bounds_x) + 0.2
            approach_y = np.mean(self.tray_center_bounds_y)
            above_tray_pos = np.array([approach_x, approach_y, 0.95])
            self.robot.move_arm_ik(above_tray_pos, eef_orientation, max_velocity=3)
            for _ in range(150):
                p.stepSimulation()
                if self.render_mode == "human":
                    time.sleep(0.01)
            # Log gripper position at z=0.95
            eef_state = p.getLinkState(self.robot.id, self.robot.eef_id)
            print(f"Gripper at above tray position, z={eef_state[0][2]:.4f}")

            # Intermediate waypoint (z=0.95)
            intermediate_tray_pos = np.array([approach_x, approach_y, 0.95])
            self.robot.move_arm_ik(intermediate_tray_pos, eef_orientation, max_velocity=3)
            for _ in range(150):
                p.stepSimulation()
                if self.render_mode == "human":
                    time.sleep(0.01)
            # Log gripper position at z=0.95
            eef_state = p.getLinkState(self.robot.id, self.robot.eef_id)
            print(f"Gripper at intermediate tray position, z={eef_state[0][2]:.4f}")

            # Lower to tray (z=0.87)
            tray_pos = np.array([approach_x, approach_y, 0.87])
            self.robot.move_arm_ik(tray_pos, eef_orientation, max_velocity=3)
            for _ in range(150):
                p.stepSimulation()
                if self.render_mode == "human":
                    time.sleep(0.01)
            # Log gripper position at z=0.87
            eef_state = p.getLinkState(self.robot.id, self.robot.eef_id)
            print(f"Gripper at release position, z={eef_state[0][2]:.4f}")

            # Check gripper height before release
            eef_state = p.getLinkState(self.robot.id, self.robot.eef_id)
            gripper_z = eef_state[0][2]
            if gripper_z <= 0.4:
                reward -= 30
                print(f"Penalty: Gripper too low before release, z={gripper_z:.4f}, reward: {reward}")

            # Check for tray collisions during lowering
            tray_contacts = []
            for link_idx in range(1, p.getNumJoints(self.robot.id)):
                contacts = p.getContactPoints(self.robot.id, self.tray_id, link_idx, -1)
                if contacts:
                    link_name = p.getJointInfo(self.robot.id, link_idx)[1].decode("utf-8")
                    tray_contacts.extend([(link_name, cp[5], cp[9]) for cp in contacts])
            cube_tray_contacts = p.getContactPoints(self.cube_id, self.tray_id)
            if tray_contacts or cube_tray_contacts:
                reward -= 20  # Penalty for tray collision
                print(f"Tray collision detected: gripper contacts: {[(name, pos, force) for name, pos, force in tray_contacts]}, cube contacts: {[(cp[5], cp[9]) for cp in cube_tray_contacts]}")
                if self.render_mode == "human":
                    p.addUserDebugText("Tray Collision", [0.5, 0.9, 0.9], textColorRGB=[1, 0, 0], textSize=2, lifeTime=0.5)

            # Pause before release to settle cube
            for _ in range(200):
                p.stepSimulation()
                if self.render_mode == "human":
                    time.sleep(0.01)

            # Release cube
            self.robot.move_gripper(0.085, max_force=100000) # Open gripper fully
            for _ in range(80):
                p.stepSimulation()
                if self.render_mode == "human":
                    time.sleep(0.01)

            # Log cube velocity post-release
            cube_vel, _ = p.getBaseVelocity(self.cube_id)
            cube_vel_magnitude = np.linalg.norm(cube_vel)
            print(f"Post-release cube velocity: {cube_vel}, magnitude: {cube_vel_magnitude:.6f}")

            # Pause to settle cube
            for _ in range(300):
                p.stepSimulation()
                if self.render_mode == "human":
                    time.sleep(0.01)

            # Reset gripper joints
            self.robot.reset_gripper_joints(self.render_mode) # Reset gripper to neutral position

            # Check if cube is in tray
            cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube_id) # Get cube position and orientation
            cube_xy = np.array(cube_pos[:2]) # Extract x, y coordinates
            cube_euler = p.getEulerFromQuaternion(cube_orn)
            distance_to_tray = np.linalg.norm(cube_xy - self.tray_pos) # Calculate distance to tray
            tray_proximity_reward = 50 * (1 - distance_to_tray / np.linalg.norm(np.array([0.7, 0.3]) - self.tray_pos)) # Calculate proximity reward
            reward += tray_proximity_reward # Add proximity reward

            # Reward/penalty for cube settling height
            if cube_pos[2] <= self.tray_max_z: # If cube is at or below tray height
                reward += 20 # Add reward
                print(f"Reward: Cube settled properly, z={cube_pos[2]:.4f}, reward: {reward}")
            elif cube_pos[2] > self.tray_max_z + 0.1: # If cube is too high
                reward -= 20 # Apply penalty
                print(f"Penalty: Cube failed to settle, z={cube_pos[2]:.4f}, reward: {reward}")

            print(f"Cube orientation: {cube_euler}")

            # Partial placement reward for extended bounds
            extended_tray_bounds_x = [0.23, 0.77] 
            extended_tray_bounds_y = [0.7, 1.1]
            extended_tray_max_z = 0.67
            if (extended_tray_bounds_x[0] <= cube_xy[0] <= extended_tray_bounds_x[1] and
                extended_tray_bounds_y[0] <= cube_xy[1] <= extended_tray_bounds_y[1] and
                cube_pos[2] <= extended_tray_max_z): # Check if within extended bounds
                reward += 50 # Add partial placement reward
                print(f"Partial placement within extended bounds: x{extended_tray_bounds_x}, y{extended_tray_bounds_y}, z<={extended_tray_max_z}")

            # Center placement reward
            if (self.tray_center_bounds_x[0] <= cube_xy[0] <= self.tray_center_bounds_x[1] and
                self.tray_center_bounds_y[0] <= cube_xy[1] <= self.tray_center_bounds_y[1] and
                cube_pos[2] <= self.tray_max_z): # Check if within center bounds
                reward += 20 # Add center placement bonus
                print(f"Center placement bonus: x{self.tray_center_bounds_x}, y{self.tray_center_bounds_y}, z<={self.tray_max_z}")

            # Full placement reward
            if (self.tray_bounds_x[0] <= cube_xy[0] <= self.tray_bounds_x[1] and
                self.tray_bounds_y[0] <= cube_xy[1] <= self.tray_bounds_y[1] and
                cube_pos[2] <= self.tray_max_z): # Check if fully within tray bounds
                reward += 200 # Add full placement reward
                self.last_is_success = True # Mark as successful
                print(f"Cube placed in tray, position: {cube_pos}, proximity reward: {tray_proximity_reward:.2f}, total reward: {reward}")
            else: # If not fully placed
                self.last_is_success = False
                print(f"Cube placement attempt, position: {cube_pos}, distance: {distance_to_tray:.4f}, tray bounds: x{self.tray_bounds_x}, y{self.tray_bounds_y}, z<={self.tray_max_z}, proximity reward: {tray_proximity_reward:.2f}, total reward: {reward}")
            if self.render_mode == "human" and (self.tray_bounds_x[0] <= cube_xy[0] <= self.tray_bounds_x[1] and
                                               self.tray_bounds_y[0] <= cube_xy[1] <= self.tray_bounds_y[1] and
                                               cube_pos[2] <= self.tray_max_z):
                p.addUserDebugText("Success Place", textColorRGB=[0, 0, 255], textPosition=[0.5, 0.9, 0.9], textSize=2, lifeTime=0.5)
                time.sleep(0.5)

            # Return to initial position
            self.robot.original_position(self.robot, render_mode=self.render_mode)
            observation, info = self.reset()
            info["terminal_observation"] = observation
            info["is_success"] = self.last_is_success
            return observation, reward, True, False, info

        elif self.current_step >= self.max_steps: # If maximum steps reached
            reward += boundary_penalty # Add boundary penalty
            observation, info = self.reset()
            info["terminal_observation"] = observation
            done = True

        observation = self.target_pos
        print(f"Step {self.current_step}: Distance difference: {distance_to_target:.4f}, eef_pos: {eef_position}, cube_pos: {cube_xy}, reward: {reward}")
        return observation, reward, done, truncated, info

    def close(self):
        # Clean up by disconnecting from PyBullet
        p.disconnect()

class UR5Robotiq85:
    def __init__(self, pos, ori): # Complete function implementation from https://github.com/leesweqq/ur5_reinforcement_learning_grasp_object/blob/main/ur5_env.py
        # Initialize robot with base position and orientation
        self.base_pos = pos # Set base position
        self.base_ori = p.getQuaternionFromEuler(ori) # Convert orientation to quaternion
        self.eef_id = 7 # End-effector link ID
        self.arm_num_dofs = 6 # Number of degrees of freedom for the arm
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0] # Resting joint positions
        self.gripper_range = [0, 0.085] # Gripper opening range (closed to fully open)
        self.max_velocity = 10 # Maximum joint velocity

    def load(self): # Complete function implementation from https://github.com/leesweqq/ur5_reinforcement_learning_grasp_object/blob/main/ur5_env.py
        # Load the URDF model of the robot
        self.id = p.loadURDF('./urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori, useFixedBase=True)
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()

        # Debug: Check finger_joint limits
        #for joint in self.joints:
        #    if joint.name == 'finger_joint':
        #        print(f"finger_joint limits: lower={joint.lowerLimit}, upper={joint.upperLimit}")

    def __parse_joint_info__(self): # Complete function implementation from https://github.com/leesweqq/ur5_reinforcement_learning_grasp_object/blob/main/ur5_env.py
        # Parse joint information into a named tuple
        jointInfo = namedtuple('jointInfo',
                               ['id', 'name', 'type', 'lowerLimit', 'upperLimit', 'maxForce', 'maxVelocity', 'controllable'])
        self.joints = [] # Initialize list of joints
        self.controllable_joints = [] # Initialize list of controllable joints

        for i in range(p.getNumJoints(self.id)): # Iterate over all joints
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(jointID)
            self.joints.append(
                jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            )
        #print(f"Parsed joints: {[j.name for j in self.joints]}")
        #if not any(name in ['finger_joint', 'left_inner_finger_joint', 'right_inner_finger_joint'] for name in [j.name for j in self.joints]):
        #    print("Warning: No gripper joints found in URDF. Check ur5_robotiq_85.urdf for correct joint names.")

        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]
        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)]

    def __setup_mimic_joints__(self): # Complete function implementation from https://github.com/leesweqq/ur5_reinforcement_learning_grasp_object/blob/main/ur5_env.py
        # Set up mimic joints to synchronize gripper movements
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {
            'right_outer_knuckle_joint': 1,
            'left_inner_knuckle_joint': 1,
            'right_inner_knuckle_joint': 1,
            'left_inner_finger_joint': -1,
            'right_inner_finger_joint': -1
        }
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0] # Get parent joint ID
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names} # Map child multipliers

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id, self.id, joint_id,
                                   jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=30000, erp=1)

    def move_gripper(self, open_length, max_force=100000): # Implementated from https://github.com/leesweqq/ur5_reinforcement_learning_grasp_object/blob/main/ur5_env.py but only change the gripper force
        # Control the gripper to open or close with a specified force
        open_length = max(self.gripper_range[0], min(open_length, self.gripper_range[1]))
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle, force=max_force)

    def move_arm_ik(self, target_pos, target_orn, max_velocity=None): # Implemented from https://github.com/leesweqq/ur5_reinforcement_learning_grasp_object/blob/main/ur5_env.py and modified
        # Move the arm to a target position using inverse kinematics
        if max_velocity is None:
            max_velocity = self.max_velocity
        joint_poses = p.calculateInverseKinematics( # Calculate joint positions
            self.id, self.eef_id, target_pos, target_orn,
            lowerLimits=self.arm_lower_limits,
            upperLimits=self.arm_upper_limits,
            jointRanges=self.arm_joint_ranges,
            restPoses=self.arm_rest_poses,
            maxNumIterations=100
        )
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i], maxVelocity=max_velocity)

    def get_current_ee_position(self): # Complete function implementation from https://github.com/leesweqq/ur5_reinforcement_learning_grasp_object/blob/main/ur5_env.py
        # Get the current end-effector position
        return p.getLinkState(self.id, self.eef_id)

    def reset_gripper_joints(self, render_mode):
        # Reset gripper joints to neutral positions
        gripper_joint_names = [
            'finger_joint',
            'left_inner_finger_joint',
            'right_inner_finger_joint',
            'left_inner_knuckle_joint',
            'right_outer_knuckle_joint',
            'right_inner_knuckle_joint'
        ]
        for joint in self.joints:
            if joint.name in gripper_joint_names and joint.controllable:
                neutral_position = 0.715 if joint.name == 'finger_joint' else 0.0
                p.setJointMotorControl2(self.id, joint.id, p.POSITION_CONTROL, targetPosition=neutral_position, force=30000)
        for _ in range(50):
            p.stepSimulation()
            if render_mode == "human":
                time.sleep(0.01)

    def original_position(self, robot, render_mode): # Implemented from https://github.com/leesweqq/ur5_reinforcement_learning_grasp_object/blob/main/ur5_env.py and modified
        # Move the robot to its original position
        target_joint_positions = [0, -1.57, 1.57, -1.5, -1.57, 0.0]
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            p.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, target_joint_positions[i])
        for _ in range(100):
            p.stepSimulation()
            if render_mode == "human":
                time.sleep(0.01)
        robot.reset_gripper_joints(render_mode)
        robot.move_gripper(0.085, max_force=100000)
        for _ in range(100):
            p.stepSimulation()
            if render_mode == "human":
                time.sleep(0.01)