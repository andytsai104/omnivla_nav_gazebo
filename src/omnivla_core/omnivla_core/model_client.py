"""Model wrapper / client.

Intended purpose:
- isolate OmniVLA-side imports from the ROS node logic
- load model + checkpoint
- expose a simple predict(...) API
"""

# TODO:
# - initialize OmniVLA / OmniVLA-edge model
# - load fine-tuned weights
# - provide predict(image, pose, prompt, goal_image=None)
