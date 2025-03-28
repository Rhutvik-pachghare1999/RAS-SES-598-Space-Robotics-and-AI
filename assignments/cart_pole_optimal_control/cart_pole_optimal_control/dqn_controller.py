import torch
import torch.nn as nn
import numpy as np

# Reuse the same DQN model definition as in your training code
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Controller with three-phase functionality and pole stabilization
class DQNController:
    def __init__(self, model, initial_state, control_delay_duration=3.0, 
                 initial_move_duration=6.0, pole_stabilize_duration=5.0, 
                 max_time=20.0, dt=0.02, pole_angle_threshold=0.1):
        """
        Args:
            model: The trained DQN model.
            initial_state: np.array of shape (4,1) representing [cart position, cart velocity, pole angle, pole angular velocity]
            control_delay_duration: Duration (seconds) for Phase 1 (initial instability, no control force)
            initial_move_duration: End time (seconds) for Phase 2 (constant cart force)
            pole_stabilize_duration: Duration (seconds) for Phase 3 (DQN control applied)
            max_time: Total simulation time (seconds)
            dt: Time step for simulation
            pole_angle_threshold: Maximum allowable absolute pole angle (radians) to ensure the pole does not fall
        """
        self.model = model
        self.x = initial_state.copy()  # system state: [cart position, cart velocity, pole angle, pole angular velocity]
        self.time_step = 0.0
        self.dt = dt
        self.control_delay_duration = control_delay_duration
        self.initial_move_duration = initial_move_duration
        self.pole_stabilize_duration = pole_stabilize_duration
        self.max_time = max_time
        self.pole_angle_threshold = pole_angle_threshold

    def step(self):
        """
        Computes the control force based on the current phase, updates the state,
        and then clamps the pole angle to keep it within safe bounds.
        
        Returns:
            force (float): The control force to be applied.
        """
        force = 0.0
        
        # Phase 1: Soft start to keep the pole upright immediately
        if self.time_step <= self.control_delay_duration:
            # Apply a small corrective force to maintain upright pole position
            pole_angle = self.x[2][0]
            if abs(pole_angle) < self.pole_angle_threshold:
                force = 0.0  # No force if the pole is upright
            else:
                # Apply small corrective forces to stabilize the pole
                force = 5.0 if pole_angle < 0 else -5.0  # Small corrective force to keep the pole upright

        # Phase 2: Apply constant force to move the cart
        elif self.time_step <= self.initial_move_duration:
            force = 15.0  # Moderate force to move cart
            # Ensure pole remains within safe limits
            self.x[2][0] = np.clip(self.x[2][0], -self.pole_angle_threshold, self.pole_angle_threshold)

        # Phase 3: Apply DQN control to stabilize the pole
        elif self.time_step <= self.initial_move_duration + self.pole_stabilize_duration:
            # Use the DQN model to select an action
            state_tensor = torch.FloatTensor(self.x.flatten()).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(self.model(state_tensor)).item()
            # Map action index to force (adjust these force values as necessary)
            force_values = [-10, -5, 0, 5, 10]
            force = force_values[action]

        # Final Phase: Stop movement and stabilize the pole completely
        else:
            force = 0.0
            self.x[0][0] = 0.0  # Halt cart movement
            self.x[2][0] = 0.0  # Ensure pole is perfectly upright

        # Safety clamp: Ensure the pole's angle remains within safe bounds
        self.x[2][0] = np.clip(self.x[2][0], -self.pole_angle_threshold, self.pole_angle_threshold)

        # Increment the simulation time
        self.time_step += self.dt
        return force

# -------------------------
# Example usage of the DQN Controller
# -------------------------
if __name__ == "__main__":
    # Load your trained DQN model (ensure the file path is correct)
    model = DQN(4, 5)
    model.load_state_dict(torch.load("/home/rhutvik/ros2_ws/src/cart_pole_optimal_control/cart_pole_optimal_control/dqn_cart_pole.pth"))
    model.eval()

    # Initialize the state: for example, zeros with shape (4,1)
    initial_state = np.zeros((4, 1))
    controller = DQNController(model, initial_state, control_delay_duration=3.0, 
                                 initial_move_duration=6.0, pole_stabilize_duration=5.0, 
                                 max_time=20.0, dt=0.02, pole_angle_threshold=0.1)

    simulation_steps = int(controller.max_time / controller.dt)
    for _ in range(simulation_steps):
        force = controller.step()
        print(f"Time: {controller.time_step:.2f}s, Force: {force:.2f}, State: {controller.x.flatten()}")
