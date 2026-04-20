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

## Export Dataset
```bash
python3 ./omnivla_finetune/export_goal_classifier_jsonl.py \
  --run-dir ./datasets/run_001 \
  --goal-library ./src/omnivla_bringup/config/goal_library.yaml \
  --out-dir ./datasets/export_goal_classifier \
  --keep-every-n 2 \
  --success-only
```

---

## PROGRESS TRACKER


### omnivla_bringup
- [x] launch/sim.launch.py
- [x] launch/data_collection.launch.py
- [x] launch/inference_nav.launch.py
- [x] launch/evaluation.launch.py
- [x] config/goal_library.yaml
- [x] config/runtime.yaml
- [x] config/collection.yaml
- [x] config/eval.yaml

### omnivla_core
- [x] inference_node.py
- [x] nav2_goal_bridge_node.py
- [x] goal_library.py
- [x] model_client.py
- [x] image_utils.py
- [x] pose_utils.py

### omnivla_data
- [x] data_logger_node.py
- [x] episode_manager_node.py
- [x] prompt_sampler.py
- [x] goal_sampler.py
- [x] sync_utils.py
- [x] export_utils.py

### omnivla_eval
- [ ] eval_runner_node.py
- [x] metrics.py
- [x] collision_monitor.py
- [x] success_checker.py
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

- [x] Nav2 baseline working (manual goal → robot moves)
- [x] goal_library.yaml defined (5–10 goals)
- [x] data logger saving valid samples
- [ ] first dataset exported
- [ ] first model trained (goal classification)
- [ ] inference node outputs correct goal ID
- [ ] Nav2 bridge works with model output
- [ ] evaluation script runs end-to-end
