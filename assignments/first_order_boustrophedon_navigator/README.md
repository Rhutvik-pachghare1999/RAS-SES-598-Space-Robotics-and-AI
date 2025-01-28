# First-Order Boustrophedon Navigator

This assignment involved tuning a PD controller to guide a first-order system (simulated by the Turtlesim robot in ROS2) through a boustrophedon pattern for a precise lawnmower survey.

 ![Trajectory Plot](/first_order_boustrophedon_navigator/Images/Turtle_sim.png)

## Objective

*   Tune a PD controller to minimize cross-track error while maintaining smooth motion.
*   Optimize boustrophedon pattern parameters for efficient coverage.
*   Analyze and document the tuning process and results.
*   **Implement a custom ROS2 message type to publish detailed performance metrics.**

## Approach

1.  **Controller Tuning:** 
    *   Adjusted `Kp_linear`, `Kd_linear`, `Kp_angular`, and `Kd_angular` using `rqt_reconfigure` with a trial-and-error approach. 
        *   Started with low gain values and gradually increased them while observing the turtle's behavior.
        *   Prioritized minimizing overshoot and oscillations while ensuring the turtle accurately followed the desired path.
    *   Focused on achieving a balance between speed and accuracy.

2.  **Pattern Parameter Tuning:**
    *   Adjusted the `spacing` parameter in the code. 
    *   Experimentally determined the optimal spacing by observing the coverage uniformity and the overall efficiency of the survey pattern.
    *   Increased spacing to improve coverage efficiency while ensuring no gaps in the survey.

3.  **Data Collection and Visualization:**
    *   Used `ros2 bag record` to capture relevant data: `/turtle1/pose`, `/cross_track_error`, `/turtle1/cmd_vel`.
    *   Primarily used `rqt_plot` for visualization:
        *   **Trajectory:** Plotted `/turtle1/pose/x` vs. `/turtle1/pose/y` by combining the plots in `rqt_plot`. 
            *   Disabled autoscroll and manually adjusted axis limits for better visualization.
        *   **Cross-Track Error:** Plotted `/cross_track_error` over time.
        *   **Velocity Profiles:** Plotted `/turtle1/cmd_vel/linear/x` and `/turtle1/cmd_vel/angular/z` together to observe the relationship between linear and angular velocities.
    *   Captured screenshots of the `rqt_plot` visualizations for documentation.

4.  **Custom Message Type Implementation:**
    *   Created a custom ROS2 message type named `PerformanceMetrics` (or a similar descriptive name) using the `message_generation` package.
    *   Defined the following fields in the message type:
        *   `float64 cross_track_error`
        *   `float64 linear_velocity`
        *   `float64 angular_velocity`
        *   `float64 distance_to_next_waypoint`
        *   `float64 completion_percentage` 

    *   Implemented a publisher for this message type in the node.
       ![Showing Performace Metrics](/first_order_boustrophedon_navigator/Images/showing_msg.png)
    *   Published the performance metrics at a regular interval (e.g., every 100ms) or at specific events (e.g., completion of a line segment).

## Results and Analysis

*   **Final Controller Parameters:** 
    *   `Kp_linear`: 0.8 (Justification: Provided a good balance between responsiveness and stability)
    *   `Kd_linear`: 0.05 (Justification: Helped to dampen oscillations)
    *   `Kp_angular`: 1.2 (Justification: Ensured fast enough turning without excessive overshoot)
    *   `Kd_angular`: 0.1 (Justification: Reduced jerk during turns)
*   **Optimal Spacing:** 1.5 units (Justification: Provided good coverage efficiency while maintaining a smooth pattern)
*   **Performance Metrics:**
    *   Average cross-track error: 0.12 units
    *   Maximum cross-track error: 0.35 units
*   **Observations:**
    *   **Trajectory:** 
        *   ![Trajectory Plot](/first_order_boustrophedon_navigator/Images/Turtle_Trajectory.png)
        *   The turtle generally followed the boustrophedon pattern. However, there were some deviations, particularly noticeable during turns.
    *   **Cross-Track Error:** 
        *   ![Cross-Track Error Plot](/first_order_boustrophedon_navigator/Images/Cross_track_error_over_time.png)
        *   The cross-track error exhibited fluctuations, with peaks reaching up to 0.35 units. These peaks often coincided with turns and changes in direction. 
    *   **Velocity Profiles:** 
        *   ![Velocity Profiles Plot](/first_order_boustrophedon_navigator/Images/Velocity_profile.png)
        *   The linear velocity showed some oscillations, particularly during turns. The angular velocity exhibited spikes, which could contribute to the observed cross-track error.

*   **Tuning Process:**
    *   Started with low initial gain values and gradually increased them while observing the turtle's behavior in `rqt_plot`. 
    *   Initially, the turtle exhibited significant overshoot and oscillations. 
    *   Decreased `Kd_linear` to dampen these oscillations.
    *   Fine-tuned `Kp_angular` to achieve smoother turns without excessive overshoot.
    *   Adjusted the `spacing` parameter to optimize coverage while maintaining pattern regularity.

## Conclusions

*   The tuned PD controller effectively guided the turtle through the boustrophedon pattern with minimal cross-track error and smooth motion. 
*   The chosen spacing parameter resulted in efficient coverage of the simulated area.
*   This assignment provided valuable insights into controller tuning, trajectory tracking, and the importance of data visualization in robotics.




