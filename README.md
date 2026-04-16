# OmniVLA Navigation (ROS 2 + Gazebo)

Hybrid language-guided navigation project using:
- ROS 2 Jazzy
- Gazebo Harmonic + BCR Bot
- Nav2 (execution)
- OmniVLA-edge (semantic goal inference)

---

## Repo Structure

```
omnivla_ws/
├── OmniVLA/                 # model + training (outside ROS)
│   ├── checkpoints/
│   ├── datasets/
│   ├── scripts/
│   └── training/
├── src/
│   ├── bcr_bot/             # third-party robot
│   ├── omnivla_bringup/     # launch + configs
│   ├── omnivla_core/        # runtime (inference + nav2 bridge)
│   ├── omnivla_data/        # data collection
│   └── omnivla_eval/        # evaluation
└── README.md
```

---

## Build

```
cd ~/ros2_projects/omnivla_ws
colcon build --symlink-install
source install/setup.bash
```

Clean rebuild if needed:

```
rm -rf build install log
colcon build --symlink-install
source install/setup.bash
```

---

## Package Responsibilities

### omnivla_bringup
- all launch files
- all yaml configs
- orchestrate modes (sim / data / inference / eval)

### omnivla_core
- inference node (OmniVLA)
- goal resolver
- Nav2 goal bridge

### omnivla_data
- data logger
- episode reset / sampling
- dataset export helpers

### omnivla_eval
- evaluation runner
- metrics (success, time, collision)
- result logging

---

## Minimal Files to Start

```
omnivla_bringup/
  launch/sim.launch.py
  launch/data_collection.launch.py
  config/goal_library.yaml

omnivla_core/
  inference_node.py
  nav2_goal_bridge_node.py
  goal_library.py

omnivla_data/
  data_logger_node.py
  episode_manager_node.py

omnivla_eval/
  eval_runner_node.py
  metrics.py
```

---

## Dev Flow

```
1. bring up sim + nav2 (bcr_bot)
2. define goal_library.yaml
3. collect data (omnivla_data)
4. export dataset → OmniVLA/
5. finetune model
6. run inference (omnivla_core)
7. send goal → Nav2
8. run evaluation (omnivla_eval)
```

---

## PROGRESS TRACKER


### omnivla_bringup
- [ ] launch/sim.launch.py
- [ ] launch/data_collection.launch.py
- [ ] launch/inference_nav.launch.py
- [ ] launch/evaluation.launch.py
- [ ] config/goal_library.yaml
- [ ] config/runtime.yaml
- [ ] config/collection.yaml
- [ ] config/eval.yaml

### omnivla_core
- [ ] inference_node.py
- [ ] nav2_goal_bridge_node.py
- [ ] goal_library.py
- [ ] model_client.py
- [ ] image_utils.py
- [ ] pose_utils.py

### omnivla_data
- [ ] data_logger_node.py
- [ ] episode_manager_node.py
- [ ] prompt_sampler.py
- [ ] goal_sampler.py
- [ ] sync_utils.py
- [ ] export_utils.py

### omnivla_eval
- [ ] eval_runner_node.py
- [ ] metrics.py
- [ ] collision_monitor.py
- [ ] success_checker.py
- [ ] result_logger.py

### OmniVLA (model side)
- [ ] dataset export script
- [ ] dataset loader
- [ ] goal classifier training
- [ ] LoRA setup
- [ ] checkpoint saving
- [ ] offline validation

---
## CURRENT MILESTONE

- [ ] Nav2 baseline working (manual goal → robot moves)
- [ ] goal_library.yaml defined (5–10 goals)
- [ ] data logger saving valid samples
- [ ] first dataset exported
- [ ] first model trained (goal classification)
- [ ] inference node outputs correct goal ID
- [ ] Nav2 bridge works with model output
- [ ] evaluation script runs end-to-end
