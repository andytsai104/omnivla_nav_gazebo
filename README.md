# OmniVLA Language-Guided Navigation (ROS 2 + Gazebo)

A modular **ROS 2 Jazzy** and **Gazebo Harmonic** project for **language-guided robot navigation**, combining **Vision-Language Models (OmniVLA-edge)** with **Nav2** for reliable execution.

This project focuses on bridging **semantic understanding (language + vision)** with **classical navigation systems**, enabling robots to interpret human commands and navigate accordingly.

![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy-blue)
![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic-orange)
![Nav2](https://img.shields.io/badge/Nav2-Navigation-green)

## Demo Video
[![Demo](./media/omnivla_demo.gif)](https://www.youtube.com/watch?v=JP31pXTPZl8)

---

## Workspace Architecture

This project follows a modular and scalable "Separation of Concerns" design:

```text
omnivla_ws/
├── OmniVLA/                  # Base VLA model (pretrained)
│
├── omnivla_finetune/         # Finetuning + classifier training
│   ├── checkpoints/
│   ├── datasets/
│   └── training scripts
│
├── src/
│   ├── bcr_bot/              # Robot + Gazebo simulation
│   │
│   ├── omnivla_bringup/      # Launch files + configs
│   │   ├── launch/
│   │   └── config/
│   │
│   ├── omnivla_core/         # Runtime system
│   │   ├── inference_node.py
│   │   ├── nav2_goal_bridge_node.py
│   │   └── model_client.py
│   │
│   ├── omnivla_data/         # Data collection pipeline
│   │   ├── episode_manager_node.py
│   │   └── data_logger_node.py
│   │
│   └── omnivla_eval/         # (optional) evaluation tools
│
├── datasets/                 # collected episodes
└── README.md
```

---

## System Pipeline

The navigation system integrates learning-based perception with classical planning:

```text
Language Prompt
        ↓
OmniVLA Inference (image + prompt)
        ↓
Predicted Goal ID
        ↓
Goal Library (ID → Pose)
        ↓
Nav2 Planner & Controller
        ↓
Robot Navigation
```

- Language → semantic goal  
- Goal → pose (via goal library)  
- Pose → execution (Nav2)  

---

## Quick Start

### 1. Prerequisites

- Ubuntu 22.04 / 24.04  
- ROS 2 Jazzy  
- Gazebo Harmonic  
- Nav2  

---

### 2. Build the Workspace

```bash
cd ~/ros2_projects/omnivla_ws
colcon build --symlink-install
source install/setup.bash
```

Clean rebuild if needed:
```bash
rm -rf build install log
colcon build --symlink-install
source install/setup.bash
```

---

### 3. Run Language-Guided Navigation

#### Launch system
```bash
ros2 launch omnivla_bringup inference_nav.launch.py
```

#### Send prompt
```bash
ros2 topic pub --once /omnivla/prompt std_msgs/msg/String "{data: 'go to the small shelf row'}"
```

---

## Data Collection Pipeline

### Start collection
```bash
ros2 launch omnivla_bringup data_collection.launch.py
```

### Control episodes
```bash
ros2 service call /omnivla_data/start_collection std_srvs/srv/Trigger
ros2 service call /omnivla_data/stop_collection std_srvs/srv/Trigger
```

---

## Dataset Export

```bash
python3 ./omnivla_finetune/export_goal_classifier_jsonl.py \
  --run-dir ./datasets/run_001 \
  --goal-library ./src/omnivla_bringup/config/goal_library.yaml \
  --out-dir ./datasets/export_goal_classifier \
  --keep-every-n 2 \
  --success-only
```

---

## Model Finetuning

```bash
python3 ./omnivla_finetune/train_omnivla_edge_classifier.py \
  --train-jsonl ./datasets/export_goal_classifier/train.jsonl \
  --val-jsonl ./datasets/export_goal_classifier/val.jsonl \
  --checkpoint-path ./omnivla-edge/omnivla-edge.pth \
  --num-classes 7 \
  --epochs 20 \
  --batch-size 4 \
  --feature-mode actions \
  --device cuda
```

Best validation accuracy achieved:

```
val_acc = 0.784
```

---

## Key Features

- Language-guided navigation  
- Vision-Language-Action (VLA) integration  
- Modular ROS2 system design  
- Automated dataset collection pipeline  
- Robust fallback (rule-based inference)  
- Easily extendable goal library  

---

## Design Highlights

- Hybrid architecture combining learning and classical planning  
- Semantic goal abstraction instead of raw pose prediction  
- Scalable data pipeline for iterative improvement  

---

## Maintainer

- **Andy Tsai**  
M.S. Robotics & Autonomous Systems @ ASU  
andystsai1040@gmail.com  
https://github.com/andytsai104  
<br>
- **Alan Cheng**
M.S. Robotics & Autonomous Systems @ ASU 
hcheng57@asu.edu
https://github.com/Ott3rAlan9Ol2S
<br>
- **Roy Yu**
M.S. Robotics & Autonomous Systems @ ASU 
mengjuyu@asu.edu
https://github.com/roy0823


---

## References & Resources

This project builds on the following upstream repositories:

- [OmniVLA](https://github.com/NHirose/OmniVLA/tree/main) for the base vision-language-action model
- [bcr_bot](https://github.com/blackcoffeerobotics/bcr_bot/tree/ros2-jazzy?tab=readme-ov-file#jazzy--harmonic-ubuntu-2404) for the robot simulation platform
- ROS 2 Navigation Stack (Nav2)  
- Gazebo Harmonic simulation  
- ROS 2 documentation  