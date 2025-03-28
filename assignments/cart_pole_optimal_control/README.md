# Cart-Pole Optimal Control Assignment

PLease find the images and videos in the end of this read me file, and i couldnt upload ros2_ws so i uploaded it here all files [https://github.com/Rhutvik-pachghare1999/RAS-SES-598-Space-Robotics-and-AI/tree/main/assignments/cart_pole_optimal_control/cart_pole_optimal_control]

## Overview
This project implements both an LQR controller and a Dueling DQN reinforcement learning agent for stabilizing a cart-pole system under earthquake disturbances. The goal is to maintain the pole's stability while ensuring the cart stays within its physical limits despite external perturbations. The earthquake force generator introduces real-world-like disturbances, making the control challenge more complex.

## System Description

### 1. LQR-Controlled Cart-Pole System

## Physical Setup
               - An inverted pendulum mounted on a cart moving along a linear track.

               - The goal is to stabilize the pendulum in an upright position while keeping the cart within operational limits.

## Key Parameters:

                 - Cart Mass: 1.0 kg

                 - Pole Mass: 1.0 kg

                 - Pole Length: 1.0 m

                 - Cart Range: ±2.5m

               - Control Strategy: Linear Quadratic Regulator (LQR)

               - State Variables: Cart position, cart velocity, pole angle, and angular velocity.

## Functionality
                     - Implements LQR-based state feedback control.

                     - Computes optimal control force to minimize deviations from the upright position.

                     - Takes into account the quadratic cost function to balance control effort and stability.


## Key Matrix:
![Screenshot from 2025-03-26 21-44-35](https://github.com/user-attachments/assets/2db7d766-803b-4ee2-8c80-a2d84dd1b05d)

### 2. Earthquake Force Generator
## Physical Setup
            - Generates external perturbations to simulate seismic disturbances acting on the cart.

            - Introduces random forces with varying amplitudes and frequencies to test controller robustness.

## Functionality
            - Uses superposition of sine waves to create realistic earthquake-like disturbances.

             - Frequency and amplitude variations introduce unpredictability in system behavior.

            - Gaussian noise adds real-world uncertainty.

## Default Parameters:
![Screenshot from 2025-03-26 21-47-17](https://github.com/user-attachments/assets/d230d566-cc82-480b-8057-f1165b0e2190)
parameters=[{
    'base_amplitude': 15.0,    # Force amplitude (N)
    'frequency_range': [0.5, 4.0],  # Frequency range (Hz)
    'update_rate': 50.0  # Update rate (Hz)
}]
### DQN Training
## Physical Setup:
- Cart-Pole System: A cart moving along a track with a pole attached, controlled by forces applied to the cart.
- State: Position and velocity of the cart, and angle and angular velocity of the pole.
- Action: 5 discrete actions, each representing a force applied to the cart.

## Functionality:
- DQN Agent: Learns to balance the pole using a neural network. It takes the state as input and outputs a force action for the cart.
- Training: The agent uses **epsilon-greedy** for exploration and exploits learned actions as it trains. **Prioritized experience replay** and **Double DQN** are used for stability.

## Default Parameters:
- Model: 3 layers with 128 neurons, Adam optimizer, MSE loss.
- Agent: 
  - Gamma: 0.99, **Epsilon**: 1.0 (decays), **Replay buffer**: 100,000, **Batch size**: 64.
  - Alpha: 0.6, **Beta**: 0.4, **Max priority**: 1.0.
  - Target Network Update: Every 100 steps.

##  Outputs:
- **Model**: Saved as `dqn_cart_pole.pth`.
- **Results**: Saved in `dqn_results.csv`.
- **Training Graphs**: Epsilon decay, rewards, and pole angle plotted.

### comparision file
  ## Physical Setup
- The script compares the performance of two control strategies for a cart-pole system:

             1) LQR (Linear Quadratic Regulator)

              2) DQN (Deep Q-Network)

- The cart-pole system consists of:

               1) A cart that moves along a track.

              2) A pole attached to the cart, which needs to be balanced.

               3) Data for each controller is stored in CSV files (lqr_data.csv and dqn_results.csv).

## Functionality
        - Reads LQR and DQN results from CSV files.

        - Ensures both datasets have the same length for fair comparison.

           -Generates plots comparing:

            a) Cart Position over time.

            b) Pole Angle over time.

             c) Control Effort (Force applied) over time.

             d) Saves the plots in /home/rhutvik/ros2_ws/build/cart_pole_optimal_control/cart_pole_optimal_control.

### Default Parameters
         a) CSV File Paths:

            - LQR results: "/home/rhutvik/ros2_ws/build/cart_pole_optimal_control/cart_pole_optimal_control/lqr_data.csv"

            - DQN results: "/home/rhutvik/ros2_ws/build/cart_pole_optimal_control/cart_pole_optimal_control/dqn_results.csv"

         b) Plot Settings:

         - Figure size: (10, 5)

         - LQR line: Dashed Blue

         - DQN line: Solid Red

          - Grid enabled

         - Horizontal reference line at y=0 (dotted black)

         - Save Directory: "/home/rhutvik/ros2_ws/build/cart_pole_optimal_control/cart_pole_optimal_control"

           c) Plot Filenames:

          - cart_position.png

          - pole_angle.png

          - control_effort.png

###  dqn_controller

## Physical Setup
  -Implements a Deep Q-Network (DQN) based controller for the cart-pole system.

-The cart-pole system consists of:

     a) Cart: Moves along a linear track.

     b) Pole: Mounted on the cart, should remain upright.

     c) State Representation (initial_state):

                                           - x[0] = Cart Position (meters)

                                           - x[1] = Cart Velocity (m/s)

                                          - x[2] = Pole Angle (radians)

                                          - x[3] = Pole Angular Velocity (rad/s)

## Functionality
a) Uses a three-phase control strategy:

                            - Phase 1 (Control Delay): Minimal force applied to prevent the pole from tipping.

                              - Phase 2 (Initial Move): Applies a constant force to move the cart.

                              - Phase 3 (DQN Control): Uses a trained DQN model to select appropriate control forces.

                              - The controller makes discrete-time updates based on time step (dt).

                              - Loads a pre-trained DQN model to make control decisions.

### Default Parameters
## Neural Network Architecture:

                                 - Input: state_dim=4 (cart position, cart velocity, pole angle, pole angular velocity)

                                  - Hidden Layers: 64, 64
        
                                    - Output: action_dim=5 (discrete force values)

## Simulation Timing:

                    - dt = 0.02 sec (time step)

                     - max_time = 20.0 sec (total simulation time)

                     - control_delay_duration = 3.0 sec (Phase 1)

                     - initial_move_duration = 6.0 sec (Phase 2)

                     - pole_stabilize_duration = 5.0 sec (Phase 3)

## Control Forces:

                   - Phase 1: 0 N (if the pole is upright) or small corrective force (±5 N).

                   - Phase 2: 15 N (to move cart).

                  - Phase 3 (DQN-based):

                      Action mapping: [-10, -5, 0, 5, 10] N.

## Pole Stability Constraint:

                           - pole_angle_threshold = 0.1 rad (Ensures pole stays within ±0.1 radians).

## DQN Model Path:

                        - "/home/rhutvik/ros2_ws/src/cart_pole_optimal_control/cart_pole_optimal_control/dqn_cart_pole.pth"



## Assignment Objectives

### Core Requirements
## 1. Analyze and Tune the Provided LQR Controller

## Objective 1: Maintain the Pendulum in an Upright Position
Current Performance: The system uses an LQR controller that has been implemented with an initial perturbation in both the cart and pole's position and velocity, ensuring that the controller works with unstable initial conditions. This helps the pendulum stabilize to the upright position over time.

# Key Tuning Insight:

a) Q Matrix: The current value for Q[2, 2] = 100.0 prioritizes the pendulum angle, which is crucial for maintaining the upright position. If we want to ensure the pendulum stays upright even with more significant disturbances or perturbations, we can increase the weight on the pendulum angle further, e.g., Q[2, 2] = 200.0. This increases the penalty for deviations in angle, encouraging the system to stabilize the pole faster.

## Objective 2: Keep the Cart Within ±2.5m Physical Limits
a) Current Performance: The cart's position is being controlled with respect to the x state (cart position). The system applies a force to stabilize the cart within its physical boundaries.

## Key Tuning Insight:

a) Q Matrix: The current penalty on the cart position (Q[0, 0] = 50.0) is already reasonable for keeping the cart within bounds. However, to prevent any major deviation in extreme cases or external disturbances, consider increasing this penalty to Q[0, 0] = 100.0 for stricter control over the cart's position.

Control Force Saturation: The force is clipped at ±20.0 (self.max_force), which ensures that the control system does not exceed the cart's capacity for force handling. This limit is essential to avoid unrealistic control actions.

## Objective 3: Achieve Stable Operation Under Earthquake Disturbances
Current Performance: The system works well in the presence of small disturbances (due to initial perturbation in the x[0] and x[2] states). However, large, sudden disturbances (like those caused by earthquakes) require more attention.

a) Key Tuning Insight:

Q Matrix: To handle external disturbances like earthquakes, the system needs to respond more robustly. Increasing the penalty on the velocity states (x_dot, theta_dot) will make the system more responsive to fast changes.


## 2. Documenting the LQR Tuning Approach

## Analysis of the Existing Q and R Matrices:
 #  a) Q Matrix:

                  The current Q matrix is defined as:
![Screenshot from 2025-03-26 22-38-55](https://github.com/user-attachments/assets/841a131b-6dc3-4dd8-ac6d-379d217427f1)

                   - Q[0, 0]: Controls the cart's position.

                    - Q[2, 2]: Controls the pendulum's angle.

                     - Q[1, 1], Q[3, 3]: Control the velocities of the cart and pendulum respectively.

- Justification: The values are chosen to prioritize the pendulum's angle more heavily (since it’s more critical to stabilize the pendulum) and provide moderate control over the cart’s position and velocities.

# b)  R Matrix:

The current R matrix is set to:
![Screenshot from 2025-03-26 22-42-21](https://github.com/user-attachments/assets/894763aa-695d-47be-abe1-87644e9cb83f)
- This is a penalty on the control force applied to the system. It is small, indicating a low penalty on control effort and allowing the system to use more control force if necessary.

## Justification for Tuning Changes:
   - Increased Pendulum Penalty: By increasing Q[2, 2] (the penalty for pendulum angle), we can ensure the pendulum is kept upright more effectively, especially under disturbances.

   - Increased Velocity Penalty: Adjusting Q[1, 1] and Q[3, 3] improves the system's ability to handle disturbances such as rapid changes in position or angle.

     - Cart Position Control: Increasing the weight on the cart's position (Q[0, 0]) ensures that the cart stays within the physical limits without large deviations.

## Analysis of Performance Trade-offs:
    - Higher Penalty on theta: Increases stability for the pendulum, but may lead to slower cart stabilization.

      - Higher Penalty on Velocities: Ensures responsiveness to disturbances but may require more control force.

       - Stronger Cart Position Penalty: Helps maintain cart within boundaries but may slow down pendulum stabilization slightly.

3. System Performance Analysis
## Duration of Stable Operation:
   - The system is designed to stabilize the cart-pole system within 50 seconds (self.max_time = 50.0). The system should ideally maintain stability throughout this period, with the pendulum remaining upright and the cart staying within 
        the ±2.5m limits.

## Maximum Cart Displacement:
    - The maximum cart displacement should ideally remain within the ±2.5m limit. By observing the system’s behavior over time, the tuning of Q[0, 0] can ensure that the cart doesn’t exceed this limit during the entire control process.

## Pendulum Angle Deviation:
 - The pendulum should ideally have minimal deviation from the upright position. The current LQR controller is designed to keep the pendulum close to zero angle. However, fine-tuning Q[2, 2] ensures faster and more reliable stabilization.

## Control Effort Analysis:
- The control effort (force applied to the cart) is capped at ±20.0. Monitoring the control force over time will provide insight into whether this limit is too high or low based on the system’s responsiveness and stability under the tuning changes.

  

### Extra Credit Options
Students can implement reinforcement learning for extra credit (up to 30 points):

### 1. Reinforcement Learning Implementation: DQN for Cart-Pole
## DQN Training Code 
 - Model Definition: The DQN model consists of a feedforward neural network with two hidden layers. It accepts a state vector (e.g., [cart position, cart velocity, pole angle, pole angular velocity]) and outputs a vector representing the Q-values for each action.

- Agent Class: The DQNAgent class is responsible for interacting with the environment and learning from experiences. It uses:

- Replay Buffer: Stores state-action-reward-next_state tuples to be used for training.

- Epsilon-Greedy Strategy: Balances exploration and exploitation. Initially, the agent explores more (high epsilon) and slowly shifts to exploitation (low epsilon).

- Double DQN: Uses two networks (the model and a target model) to avoid overestimation of Q-values, stabilizing learning.

- Prioritized Experience Replay: Prioritizes more important experiences based on temporal difference (TD) error, enabling the agent to learn from more impactful experiences.

- Gradient Clipping: To prevent exploding gradients during training.

## DQN Training Loop
- Data Preparation: The lqr_data.csv file contains optimal control data for training the agent. The data includes state vectors, actions, rewards, and next states.

- Training Process: The agent's training loop iterates over the state data, storing experiences and updating the model using the training method in DQNAgent. The model is saved periodically after training and the results are saved in a CSV file.

- Visualization: Training progress is visualized in graphs showing:

- Epsilon Decay Over Time (exploration vs. exploitation).

- Rewards Over Time (how much reward the agent accumulates).

- Pole Angle (Theta) Over Time (stabilization of the pole).

  ![Screenshot from 2025-03-26 23-09-27](https://github.com/user-attachments/assets/d48153fd-1e0a-420b-a394-7796a43ec58b)


### 2. Compare Performance with LQR Controller
### Comparison File

 - Data Loading: The LQR results (lqr_data.csv) and the DQN results (dqn_results.csv) are loaded for comparison.

 - Plotting: Graphs are generated to compare the performance of the DQN agent with the LQR controller across several metrics:

- Cart Position: Comparing the cart's movement over time.
  ![Screenshot from 2025-03-26 23-04-24](https://github.com/user-attachments/assets/6923e9c2-36a6-48ea-844d-67af4dcc0565)

- Pole Angle: Comparing how well the pole is stabilized by both controllers.
![Screenshot from 2025-03-26 23-05-05](https://github.com/user-attachments/assets/8f69079c-7c42-440a-908e-25695d6187c6)

-  Control Force: Comparing the force applied by both controllers over time.
![Screenshot from 2025-03-26 23-06-29](https://github.com/user-attachments/assets/4ff49950-f574-4804-80e7-1cab005765e5)

The code visualizes these comparisons in three different plots, each comparing the respective values from the LQR and DQN controller results over time.

### 3. Stabilizing the Pendulum Using DQN

## DQN Controller 

- Controller Design: The DQNController class uses the trained DQN model to control the cart-pole system. It operates in three phases:

        a) Phase 1 (Initial Instability): No control force is applied immediately, allowing the pendulum to become unstable.

       b) Phase 2 (Cart Movement): A constant force is applied to move the cart.

        c) Phase 3 (Pole Stabilization): The DQN model is used to control the pole stabilization by selecting actions based on the current state of the system.

        d) Final Phase: The movement is stopped and the pole is stabilized completely.

- The step method calculates the control force for each time step based on the current state and phase of the control process. The state is updated at each step, and the force is applied accordingly.

### 4. Training Process & Results Documentation
The training process is recorded in both CSV files (dqn_results.csv and lqr_data.csv) and graphs (saved as PNG images), which visualize the learning progression and the comparison of the DQN agent with the LQR controller.

## How it Works in Context:
  - The goal is to train a reinforcement learning agent (DQN) to perform better than an optimal control strategy (LQR) for stabilizing the cart-pole system.

  - In the training phase, the DQN model is fed data from the environment (state-action-reward-next_state), and the model learns to predict the best actions (forces) to apply to the cart-pole system.

- After training, the performance of the DQN agent is compared to the LQR controller, and the results are visualized to evaluate which method performs better in stabilizing the pendulum (pole).



### Learning Outcomes
- Understanding of LQR control parameters and their effects
- Experience with competing control objectives
- Analysis of system behavior under disturbances
- Practical experience with ROS2 and Gazebo simulation
- Deep Q-Networks (DQN) Understanding: Learn the structure and application of DQNs for reinforcement learning, including Double DQN and prioritized experience replay.
- Training Reinforcement Learning Agents: Gain experience in training RL agents, using epsilon decay, loss functions, and optimizers to improve performance.
 -Optimal Control Comparison: Understand how to compare RL-based and LQR control strategies by analyzing performance metrics like cart position and pole angle.
- Real-World RL Applications: Apply RL to control dynamic systems, manage safety constraints, and adapt control strategies over time.
- Practical RL Tools: Get hands-on experience using PyTorch for DQN implementation and handling large datasets in RL.
- Model Improvement and Debugging: Learn to tune hyperparameters, stabilize training, and evaluate RL models effectively.
- 
## Implementation

### LQR Controller Description
The package includes a complete LQR controller implementation (`lqr_controller.py`) with the following features:
- State feedback control
- Configurable Q and R matrices
- Real-time force command generation
- State estimation and processing

### Training Description
- **Model**: DQN with three fully connected layers.
- **Replay Buffer**: Stores experiences; samples for training.
- **Double DQN**: Uses a target network to reduce Q-value bias.
- **Exploration**: Epsilon-greedy, decays over time.
- **Loss**: Mean squared error (MSE) between predicted and target Q-values.
- **Optimizer**: Adam to minimize the loss.
- **Saving**: Trained model is saved after training.

### DQN Controller Description
- **State**: Cart position, velocity, pole angle, angular velocity.
- **Control**: Force applied based on state (DQN model output).
- **Phases**: 
  - Phase 1: Corrective force.
  - Phase 2: Constant force to move the cart.
  - Phase 3: DQN-selected force for stabilization.
- **Safety**: Pole angle clamped to safe bounds.
- **Integration**: Trained model used for real-time control.
  

## Getting Started

### Prerequisites
- ROS2 Humble or Jazzy
- Gazebo Garden
- Python 3.8+
- Required Python packages: numpy, scipy, torch, matplotlib

#### Installation Commands
```bash
# Set ROS_DISTRO as per your configuration
export ROS_DISTRO=humble

# Install ROS2 packages
sudo apt update
sudo apt install -y \
    ros-$ROS_DISTRO-ros-gz-bridge \
    ros-$ROS_DISTRO-ros-gz-sim \
    ros-$ROS_DISTRO-ros-gz-interfaces \
    ros-$ROS_DISTRO-robot-state-publisher \
    ros-$ROS_DISTRO-rviz2

# Install Python dependencies
pip3 install numpy scipy control
```

### Repository Setup

#### If you already have a fork of the course repository:
```bash
# Navigate to your local copy of the repository
cd ~/RAS-SES-598-Space-Robotics-and-AI

# Add the original repository as upstream (if not already done)
git remote add upstream https://github.com/DREAMS-lab/RAS-SES-598-Space-Robotics-and-AI.git

# Fetch the latest changes from upstream
git fetch upstream

# Checkout your main branch
git checkout main

# Merge upstream changes
git merge upstream/main

# Push the updates to your fork
git push origin main
```

#### If you don't have a fork yet:
1. Fork the course repository:
   - Visit: https://github.com/DREAMS-lab/RAS-SES-598-Space-Robotics-and-AI
   - Click "Fork" in the top-right corner
   - Select your GitHub account as the destination

2. Clone your fork:
```bash
cd ~/
git clone https://github.com/YOUR_USERNAME/RAS-SES-598-Space-Robotics-and-AI.git
```

### Create Symlink to ROS2 Workspace
```bash
# Create symlink in your ROS2 workspace
cd ~/ros2_ws/src
ln -s ~/RAS-SES-598-Space-Robotics-and-AI/assignments/cart_pole_optimal_control .
```

### Building and Running
```bash
# Build the package
cd ~/ros2_ws
colcon build --packages-select cart_pole_optimal_control --symlink-install

# Source the workspace
source install/setup.bash

# Launch the simulation with visualization
ros2 launch cart_pole_optimal_control cart_pole_rviz.launch.py
```


This will start:
- Gazebo simulation (headless mode)
- RViz visualization showing:
  * Cart-pole system
  * Force arrows (control and disturbance forces)
  * TF frames for system state
 
Steps how to run 
- LQR controller[run it with commands]
- run DQN Training[run it in visual code to get the .pth model file]
- run Comparision File [run it in visual code to get comparsion charts]
- comment lqr controller and uncomment dqn controller, run dqn_controller [run it with commands]

### Visualization Features
The RViz view provides a side perspective of the cart-pole system with:

#### Force Arrows
Two types of forces are visualized:
1. Control Forces (at cart level):
   - Red arrows: Positive control force (right)
   - Blue arrows: Negative control force (left)

2. Earthquake Disturbances (above cart):
   - Orange arrows: Positive disturbance (right)
   - Purple arrows: Negative disturbance (left)

Arrow lengths are proportional to force magnitudes.

## Analysis Requirements

### Performance Metrics
1. Stability Metrics:
   - Maximum pole angle deviation
   - RMS cart position error
   - Peak control force used
   - Recovery time after disturbances

2. System Constraints:
   - Cart position limit: ±2.5m
   - Control rate: 50Hz
   - Pole angle stability
   - Control effort efficiency

### Analysis Guidelines
1. Baseline Performance:
   - Document system behavior with default parameters
   - Identify key performance bottlenecks
   - Analyze disturbance effects

2. Parameter Effects:
   - Analyze how Q matrix weights affect different states
   - Study R value's impact on control aggressiveness
   - Document trade-offs between objectives

3. Disturbance Response:
   - Characterize system response to different disturbance frequencies
   - Analyze recovery behavior
   - Study control effort distribution
  
   ### Performance Metrics and Analysis Guidelines

#### 1. DQN Training Code
   **Metrics**:
   - **Training Loss**: Track the loss during training to ensure the model converges.
   - **Epsilon Decay**: Monitor how epsilon decreases over time, ensuring gradual exploration reduction.
   - **Rewards History**: Track the rewards obtained during training to assess the agent's learning progress.
   - **Pole Angle (Theta)**: Measure the stability of the pole by tracking its angle over time.

   **Analysis**:
   - **Loss Convergence**: Check if the loss decreases steadily, indicating that the model is learning effectively.
   - **Exploration vs Exploitation**: Ensure epsilon decays appropriately, balancing exploration and exploitation.
   - **Stability**: Evaluate how well the rewards and pole angle converge to a stable, minimal loss state, indicating effective pole stabilization.

#### 2. **Comparison Code**
   **Metrics**:
   - **Cart Position (x)**: Compare the cart's position over time for both LQR and DQN.
   - **Pole Angle (Theta)**: Compare the angle of the pole for both controllers.
   - **Control Effort (Force)**: Compare the control force exerted by the LQR and DQN.

   **Analysis**:
   - **Performance**: Compare how well the DQN matches or outperforms the LQR in stabilizing the pole and moving the cart.
   - **Control Effort**: Analyze whether the DQN requires more or less control effort compared to LQR.
   - **Stability**: Assess how quickly each controller stabilizes the pole and cart position.

#### 3. **DQN Controller Code**
   **Metrics**:
   - **Force Applied**: Measure the control force applied during each phase.
   - **Pole Angle (Theta)**: Track the pole angle to ensure the pole remains upright within safe bounds.
   - **Simulation Time**: Evaluate how long the controller takes to stabilize the system.

   **Analysis**:
   - **Phase Transition**: Ensure smooth transitions between phases (initial instability, cart movement, and pole stabilization).
   - **Pole Stability**: Verify the pole remains within the desired angle thresholds, reflecting effective control.
   - **Time Efficiency**: Evaluate how quickly the system stabilizes and if the controller achieves the goal within the defined time.

#### 4. **Overall System Performance**
   **Metrics**:
   - **Total Rewards**: Sum of rewards over time, indicating overall system performance.
   - **Average Control Force**: Mean force applied during control, representing efficiency.
   - **Pole Angle Deviation**: Deviation from the upright position, indicating stability.
   - **Time to Stabilize**: Time taken to bring the system to a stable state.

   **Analysis**:
   - **Efficiency**: Compare the efficiency of DQN vs LQR in terms of time and control effort.
   - **Effectiveness**: Ensure that both systems can stabilize the pole and move the cart to the desired position.
   - **Adaptability**: Assess how well the DQN adapts to different initial conditions or disturbances compared to LQR.

Model: 
 Find it in directory as it cant be upload here.
 
 LQR RUNNING MODEL AFTER STABILIZATION :
    SCREENSHOT:  
    [Screenshot from 2025-03-27 21-29-06](https://github.com/user-attachments/assets/046ee85c-a380-45a1-88c0-6f93033229c7)

  LQR MODEL Trying to STABILIZE
     VIDEO: 
     [Screencast from 2025-03-27 21-30-31.webm](https://github.com/user-attachments/assets/591d7acd-a730-4ec2-ace3-df6216521825)

Dqn after STABILIZATION:
    Screenshot :
    [Screenshot from 2025-03-27 21-38-36](https://github.com/user-attachments/assets/ef65838b-927f-4bb7-9cb5-9d6c47aae91b)
    
DQL MODEL Trying to stabilize: 
      Video:    
[Screencast from 2025-03-27 21-32-08.webm](https://github.com/user-attachments/assets/9516eaf4-01ad-4d7c-bb43-97ca759b387b)

     REFER THE .TXT  FILE FOR MORE INFORMATION 



## License
This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
[![Creative Commons License](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/) 
