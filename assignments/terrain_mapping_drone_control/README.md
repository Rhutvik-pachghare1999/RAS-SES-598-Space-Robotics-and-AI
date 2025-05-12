
# 🛰️ Mission Brief: Autonomous Cylinder Analysis and Precision Marker Landing

This mission blends two critical capabilities of autonomous drones: estimating the size and position of cylindrical formations using a circular search pattern and executing an accurate landing using visual cues from ArUco markers through Image-Based Visual Servoing (IBVS).

---

## 🚀 Mission Workflow

### 🔧 Step 1: Initialize the Simulation
Launch the simulation environment with:
```bash
ros2 launch mission cylinder_landing.launch.py
```

### 🎯 Step 2: Start Marker Detection
Activate the ArUco marker tracker with:
```bash
ros2 run mission aruco_tracker.py
```

### 🧠 Step 3: Begin the Mission
Execute the autonomous behavior script:
```bash
ros2 run mission Mission.py
```

---

## 🧭 Mission Stages

### 🌀 Stage 1: Circular Cylinder Estimation

1. **Takeoff and Mode Setup**: The drone auto-arms and switches to OFFBOARD mode.
2. **Ascend to Initial Altitude**: It vertically climbs to 5 meters below the world origin.
3. **Transition to Search Radius**: It flies to a position 15 meters ahead and begins circling.
4. **Circular Trajectory Execution**: The drone flies counter-clockwise around a 15-meter radius path while continuously scanning the terrain using its RGB and depth cameras.
5. **Cylinder Detection**: During flight, the drone analyzes the depth and size of cylindrical objects.
6. **Pause for Analysis**: After one complete loop, it hovers for a few seconds to finalize cylinder analysis.

---

### 🎯 Stage 2: Visual Marker-Based Landing

1. **Descend for Marker Search**: The drone moves to a lower altitude to prepare for marker detection.
2. **Scan First Marker**: It hovers near one potential landing marker and records its position.
3. **Center Reset**: It returns to the center of the two marker regions.
4. **Scan Second Marker**: It flies to the opposite side, hovers again, and records the second marker's data.
5. **Marker Comparison**: Using depth information, it determines which marker is closer and more suitable for landing.
6. **Approach Adjustment**: The drone adjusts its position to align better with the chosen marker.
7. **Visual Servoing for Precision Landing**: Leveraging image feedback, the drone gradually minimizes its position error until perfectly aligned for landing.

---

## 🧪 Mission Summary

This project demonstrates the integration of autonomous flight planning, depth-based object estimation, and visual servoing. The drone performs high-level spatial understanding through circular flight while achieving low-level precision through marker-based landing techniques.


---

## 📸 Visual Documentation

### 🧠 Cylinder Estimation  

![image](https://github.com/user-attachments/assets/48fdb537-0653-46f5-96b5-62bef5003303)
![ESTIMATION1](https://github.com/user-attachments/assets/ae678afe-e3bc-422f-a0be-043c7a566f5c)

---

### 🎯 ArUco Marker Detection  
![TRACKING](https://github.com/user-attachments/assets/f587558f-0997-4d3b-8e8a-b34e9055bdc1)  

---

### 🛬 Landing Sequence  
![image](https://github.com/user-attachments/assets/a4e7c468-1f92-4ae7-aad8-881e38c499d6)
![CHECKING1](https://github.com/user-attachments/assets/7d9d1ca3-6bb7-4173-b4bb-2e104f8b7c7c)  
![CHECKING](https://github.com/user-attachments/assets/0a4dda2a-dcd5-4eae-a4fa-b01e49976a5e)  
![LANDING POINT](https://github.com/user-attachments/assets/8c328b20-49ff-4f7f-a642-e1304386bd76)  
![image](https://github.com/user-attachments/assets/076dd5d3-18ce-4dd9-90df-ef25349cae9f)
