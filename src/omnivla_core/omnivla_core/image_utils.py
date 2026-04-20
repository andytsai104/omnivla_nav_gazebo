"""Image utilities.

Intended purpose:
- convert ROS image messages
- resize / normalize inputs for the model
- keep image preprocessing logic out of the main node
"""

#!/usr/bin/env python3
from __future__ import annotations

import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

_bridge = CvBridge()


def ros_image_to_bgr(msg: Image) -> np.ndarray:
    return _bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def preprocess_image(
    img_bgr: np.ndarray,
    width: int = 224,
    height: int = 224,
) -> np.ndarray:
    img = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA)
    img = bgr_to_rgb(img).astype(np.float32) / 255.0
    return img