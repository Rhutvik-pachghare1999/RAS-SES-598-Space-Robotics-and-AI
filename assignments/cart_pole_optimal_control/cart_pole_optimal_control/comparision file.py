import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
lqr_df = pd.read_csv("/home/rhutvik/ros2_ws/build/cart_pole_optimal_control/cart_pole_optimal_control/lqr_data.csv")
dqn_df = pd.read_csv("/home/rhutvik/ros2_ws/build/cart_pole_optimal_control/cart_pole_optimal_control/dqn_results.csv")

# Ensure both data have the same length for fair comparison
min_len = min(len(lqr_df), len(dqn_df))
lqr_df = lqr_df.iloc[:min_len]
dqn_df = dqn_df.iloc[:min_len]

# Time steps
time_steps = np.arange(min_len)

# Save directory
save_dir = "/home/rhutvik/ros2_ws/build/cart_pole_optimal_control/cart_pole_optimal_control"

# Function to plot and save graphs
def plot_and_save(x_data, y_lqr, y_dqn, ylabel, title, filename, lqr_label="LQR", dqn_label="DQN"):
    plt.figure(figsize=(10, 5))
    plt.plot(x_data, y_lqr, label=lqr_label, linestyle="dashed", color="blue")
    plt.plot(x_data, y_dqn, label=dqn_label, color="red")
    plt.axhline(0, color="black", linestyle="dotted")
    plt.xlabel("Time Steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    
    # Save the plot
    save_path = save_dir + filename
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

# Plot and save cart position
plot_and_save(time_steps, lqr_df["x"], dqn_df["x"], 
              ylabel="Cart Position (m)", 
              title="Cart Position Over Time", 
              filename="cart_position.png")

# Plot and save pole angle
plot_and_save(time_steps, lqr_df["theta"], dqn_df["theta"], 
              ylabel="Pole Angle (rad)", 
              title="Pole Angle Over Time", 
              filename="pole_angle.png")

# Plot and save control force
plot_and_save(time_steps, lqr_df["control_force"], dqn_df["action"], 
              ylabel="Control Force (N)", 
              title="Control Effort Over Time", 
              filename="control_effort.png")
