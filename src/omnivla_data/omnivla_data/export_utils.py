#!/usr/bin/env python3
"""
export_utils.py
---------------
Converts the collected raw ROS dataset (JSON + PNG) into a standard HDF5 format
suitable for training Vision-Language-Action (VLA) models.
"""

import os
import json
import numpy as np
from pathlib import Path

try:
    import h5py
    import cv2
    H5_AVAILABLE = True
except ImportError:
    H5_AVAILABLE = False


class DatasetExporter:
    def __init__(self, raw_dataset_dir: str, output_h5_path: str):
        """
        :param raw_dataset_dir: Path to the collected run directory (e.g., 'datasets/run_001')
        :param output_h5_path: Path where the output .h5 file will be saved.
        """
        self.raw_dir = Path(raw_dataset_dir)
        self.out_path = Path(output_h5_path)

    def export_to_hdf5(self):
        if not H5_AVAILABLE:
            print("[Error] Missing dependencies. Please run: pip install h5py opencv-python")
            return False

        if not self.raw_dir.exists():
            print(f"[Error] Raw dataset directory not found: {self.raw_dir}")
            return False

        # Find all valid episode directories
        episodes = sorted([d for d in self.raw_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
        if not episodes:
            print("[Warning] No episodes found to export.")
            return False

        print(f"Found {len(episodes)} episodes. Starting HDF5 export to {self.out_path}...")

        # Create output directory if it doesn't exist
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(self.out_path, 'w') as h5f:
            for ep_path in episodes:
                self._process_episode(ep_path, h5f)
                
        print("\nExport completed successfully!")
        return True

    def _process_episode(self, ep_path: Path, h5f: h5py.File):
        ep_name = ep_path.name
        meta_path = ep_path / "metadata.json"
        frames_dir = ep_path / "frames"
        
        # Skip incomplete episodes
        if not meta_path.exists() or not frames_dir.exists():
            print(f"  Skipping {ep_name} (Missing metadata or frames folder)")
            return

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # Only export successful episodes (optional, you can modify this based on needs)
        if meta.get("outcome") != "success":
            print(f"  Skipping {ep_name} (Outcome: {meta.get('outcome')})")
            return

        print(f"  Processing {ep_name}...")
        
        # Create a group for this episode
        ep_grp = h5f.create_group(ep_name)
        
        # Save episode-level text metadata
        ep_grp.attrs['goal_id'] = meta.get('goal_id', '')
        ep_grp.attrs['prompt'] = meta.get('prompt', '')
        
        # Lists to gather frame data
        images = []
        positions = []
        yaws = []
        timestamps = []

        frame_files = sorted([f for f in frames_dir.iterdir() if f.suffix == '.json'])
        
        for frame_json_path in frame_files:
            with open(frame_json_path, 'r') as f:
                frame_data = json.load(f)
                
            frame_idx = frame_data["frame_idx"]
            img_path = frames_dir / f"{frame_idx:04d}.png"
            
            if not img_path.exists():
                continue
                
            # Read image and convert to RGB
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                
                # Extract odometry data
                odom = frame_data["odom"]
                positions.append([odom["position"]["x"], odom["position"]["y"]])
                yaws.append(odom["yaw_rad"])
                timestamps.append(frame_data["ros_stamp"])

        # Convert to numpy arrays and save into HDF5 datasets
        if images:
            ep_grp.create_dataset('images', data=np.array(images, dtype=np.uint8), compression='gzip')
            ep_grp.create_dataset('positions', data=np.array(positions, dtype=np.float32))
            ep_grp.create_dataset('yaws', data=np.array(yaws, dtype=np.float32))
            ep_grp.create_dataset('timestamps', data=np.array(timestamps, dtype=np.float64))

# ---------------------------------------------------------------------------
# Simple CLI for testing the export
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Export OmniVLA dataset to HDF5")
    parser.add_argument('--input', type=str, default='datasets/run_001', help='Raw dataset directory')
    parser.add_argument('--output', type=str, default='datasets/export/omnivla_dataset.h5', help='Output HDF5 path')
    args = parser.parse_args()
    
    exporter = DatasetExporter(args.input, args.output)
    exporter.export_to_hdf5()