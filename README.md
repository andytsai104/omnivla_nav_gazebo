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


## Test inference_nav.launch.py
terminl 1:
```bash
ros2 launch omnivla_bringup inference_nav.launch.py
```
terminal 2:
```bash
ros2 topic pub --once /omnivla/prompt std_msgs/msg/String "{data: 'go in the small shelf row'}"
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

## Fintune Omnivla-edge
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

## Notes:
best val_acc: 0.56
```bash
python3 ./omnivla_finetune/train_omnivla_edge_classifier.py \
--train-jsonl ./datasets/export_goal_classifier/train.jsonl \
--val-jsonl ./datasets/export_goal_classifier/val.jsonl \
--checkpoint-path ./omnivla-edge/omnivla-edge.pth \
--num-classes 7 \
--epochs 20 \
--batch-size 4 \
--feature-mode actions \
--device cuda \
--lr 1e-4 \

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
- [x] dataset export script
- [x] dataset loader
- [x] goal classifier training
- [ ] LoRA setup
- [ ] checkpoint saving
- [ ] offline validation

---
## CURRENT MILESTONE

- [x] Nav2 baseline working (manual goal → robot moves)
- [x] goal_library.yaml defined (5–10 goals)
- [x] data logger saving valid samples
- [x] first dataset exported
- [x] first model trained (goal classification)
- [ ] inference node outputs correct goal ID
- [ ] Nav2 bridge works with model output
- [ ] evaluation script runs end-to-end
