#!/usr/bin/env python3
"""
Script to convert AIRBOT bson data to LeRobot dataset.
Assert arms are in left-right order and cams correspond with cam_mapping.

Example usage: python convert_bson_to_lerobot.py --bson-dir /path/to/bson/data --repo-id <org>/<dataset-name> [--no-push-to-hub]
"""

import dataclasses
import os
from pathlib import Path
import shutil
from typing import Literal
from io import BytesIO

import cv2
import numpy as np
import tqdm
import tyro
import json
import bson # uv pip install pymongo
import av

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

LEROBOT_HOME = Path(os.getenv("LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def find_bson_dirs(dir_path):
    """Find all directories containing bson files."""
    result = []

    for root, dirs, files in os.walk(dir_path):
        for dir_name in dirs:
            full_dir = Path(root) / dir_name
            bson_file = full_dir / "data.bson"
            if bson_file.exists():
                result.append(str(full_dir))

    return sorted(result)


def load_bson(bson_path):
    """Load a bson file and return its contents."""
    with open(bson_path, "rb") as f:
        data = bson.decode(f.read())
    return data


def decode_h264(h264_bytes):
    """Decode H264 video bytes into a list of frames."""
    inbuf = BytesIO(h264_bytes)
    container = av.open(inbuf)
    ret = [{
        "t": int(frame.pts * frame.time_base * 1e3),
        "data": frame.to_ndarray(format="bgr24")
    } for frame in container.decode(video=0)]
    assert len(ret) > 0, "No frames found in h264"
    return ret


def create_empty_dataset(
    repo_id: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    """Create an empty lerobot dataset with the appropriate features."""
    # in left-right order
    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "observation.velocity": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "observation.effort": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        # 当前task还有几帧结束（包括当前帧）
        "task_len": {
            "dtype": "int32",
            "shape": (1,),
            "names": [
                "len"
            ]
        }
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=25,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def extract_images_from_bson(bson_data):
    """Extract images from bson data."""
    imgs_per_cam = {}
    cam_mapping = {
        "cam1": "cam_high",
        "cam2": "cam_left_wrist",
        "cam3": "cam_right_wrist"
    }
    
    # Find all camera keys in the data
    cam_keys = [k for k in bson_data["data"] if "images" in k]
    
    # Initialize with empty arrays for all cameras
    for cam_name in cam_mapping.values():
        imgs_per_cam[cam_name] = []
    
    # Extract images for each camera
    for cam_key in cam_keys:
        cam_index = cam_key.split("/")[-1]  # Extract cam1, cam2, etc.
        if cam_index in cam_mapping:
            lerobot_cam_name = cam_mapping[cam_index]
            frames = bson_data["data"][cam_key]
            
            # Check if the frames are encoded as H264 video
            if isinstance(frames, bytes):
                # Decode the H264 video
                decoded_frames = decode_h264(frames)
                # Extract the image data
                imgs_per_cam[lerobot_cam_name] = [frame["data"] for frame in decoded_frames]
            else:
                # Process individual frames
                for frame in frames:
                    # Convert the image data to a numpy array
                    img_data = frame["data"]
                    if isinstance(img_data, bytes):
                        # If the image is encoded, decode it
                        nparr = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    else:
                        # If the image is already a numpy array
                        img = np.array(img_data)
                    
                    # Convert BGR to RGB
                    if img.ndim == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    imgs_per_cam[lerobot_cam_name].append(img)
    
    # Make sure all cameras have the same number of frames
    # If a camera has no frames, create empty frames
    frame_counts = [len(frames) for frames in imgs_per_cam.values()]
    if not all(count == frame_counts[0] for count in frame_counts) or 0 in frame_counts:
        raise ValueError(f"Inconsistent number of frames across cameras: {frame_counts}")
    
    return {k: np.array(v) for k, v in imgs_per_cam.items()}


class DefaultPrompt:
    """A class that returns the default prompt for any index."""
    def __init__(self, default_prompt):
        self.default_prompt = default_prompt
    
    def __getitem__(self, index):
        return self.default_prompt

class LongTaskLen:
    def __getitem__(self, index):
        return 1000 # longer than any action horizon

def extract_task_prompt(additional_file, ep_idx, default_prompt):
    """Extract task prompt from bson data.
    
    Args:
        additional_file: Path to the additional.json file
        ep_idx: Episode index/key in the additional data
        default_prompt: Default prompt to use if no task prompts are available
    
    Returns:
        A list of task prompts for each frame or a DefaultPrompt object
    """
    # Try to load additional data
    additional_data = None
    if additional_file.exists():
        with open(additional_file, 'r') as f:
            additional_data_all = json.load(f)
        if ep_idx in additional_data_all:
            additional_data = additional_data_all[ep_idx]
    
    # If no additional data, return default prompt
    if additional_data is None:
        print(f"use default prompt {default_prompt} for ep {ep_idx}, not exist")
        return DefaultPrompt(default_prompt)#, LongTaskLen()
    
    split_frame_indices = additional_data.get("marked_frames", [])
    task_promts = additional_data.get("task_prompts", [])
    
    # Create a list to hold the task prompts for each frame
    task_prompt = []
    task_len = []
    start_idx = -1
    for task_idx, end_idx in enumerate(split_frame_indices):
        # Get the task prompt for the current segment
        current_task_prompt = task_promts[task_idx]
        task_prompt.extend([current_task_prompt] * (end_idx - start_idx))
        task_len.extend(list(range(end_idx - start_idx, 0, -1)))
        start_idx = end_idx
    
    return task_prompt, task_len


def extract_state_and_action_from_bson(bson_data):
    """Extract state and action data from bson data."""
    # Find arm and eef keys
    arm_keys = [k for k in bson_data["data"] if "arm" in k]
    eef_keys = [k for k in bson_data["data"] if "eef" in k]
    
    # Separate observation and action keys
    obs_arm_keys = [k for k in arm_keys if "observation" in k]
    act_arm_keys = [k for k in arm_keys if "action" in k]
    obs_eef_keys = [k for k in eef_keys if "observation" in k and "joint_position" in k]
    act_eef_keys = [k for k in eef_keys if "action" in k and "joint_position" in k]
    
    # Verify that we have both left and right arm data
    left_obs_arm_key = next((k for k in obs_arm_keys if "observation1" in k), None)
    right_obs_arm_key = next((k for k in obs_arm_keys if "observation2" in k), None)
    left_act_arm_key = next((k for k in act_arm_keys if "action1" in k), None)
    right_act_arm_key = next((k for k in act_arm_keys if "action2" in k), None)
    left_obs_eef_key = next((k for k in obs_eef_keys if "observation1" in k), None)
    right_obs_eef_key = next((k for k in obs_eef_keys if "observation2" in k), None)
    left_act_eef_key = next((k for k in act_eef_keys if "action1" in k), None)
    right_act_eef_key = next((k for k in act_eef_keys if "action2" in k), None)
    
    # Ensure all required keys are present
    if not (left_obs_arm_key and right_obs_arm_key and left_act_arm_key and right_act_arm_key 
            and left_obs_eef_key and right_obs_eef_key and left_act_eef_key and right_act_eef_key):
        raise ValueError(f"Missing required keys for dual-arm data. Found: {arm_keys + eef_keys}")
    
    # Get the number of frames
    frame_num = len(bson_data["data"][left_obs_arm_key])
    
    # Initialize state and action arrays
    state = np.zeros((frame_num, 14), dtype=np.float32)  # 14 motors (7 for each arm)
    velocity = np.zeros((frame_num, 14), dtype=np.float32)
    effort = np.zeros((frame_num, 14), dtype=np.float32)
    
    action = np.zeros((frame_num, 14), dtype=np.float32)
    
    for modality, name in zip([state, velocity, effort], ["pos", "vel", "eff"]):
        # Extract joint positions and gripper data
        for i in range(frame_num):
            # Extract joint positions for left arm observation
            modality[i, 0:6] = bson_data["data"][left_obs_arm_key][i]["data"][name]
            
            # Extract joint positions for right arm observation
            modality[i, 7:13] = bson_data["data"][right_obs_arm_key][i]["data"][name]
            
            # Extract gripper position for left arm observation
            modality[i, 6:7] = bson_data["data"][left_obs_eef_key][i]["data"][name]

            # Extract gripper position for right arm observation
            modality[i, 13:14] = bson_data["data"][right_obs_eef_key][i]["data"][name]
            
    for i in range(frame_num):
        # Extract joint positions for left arm action
        action[i, 0:6] = bson_data["data"][left_act_arm_key][i]["data"]["pos"]
        
        # Extract joint positions for right arm action
        action[i, 7:13] = bson_data["data"][right_act_arm_key][i]["data"]["pos"]
        
        # Extract gripper position for left arm action
        action[i, 6:7] = bson_data["data"][left_act_eef_key][i]["data"]["pos"]
        
        # Extract gripper position for right arm action
        action[i, 13:14] = bson_data["data"][right_act_eef_key][i]["data"]["pos"]
    
    return state, velocity, effort, action


def load_bson_episode_data(ep_path, default_prompt):
    """Load episode data from a bson file."""
    bson_file = Path(ep_path) / "data.bson"
    additional_file = Path(ep_path).parent / "additional.json"
    
    # Load bson data
    bson_data = load_bson(bson_file)

    # Extract images
    imgs_per_cam = extract_images_from_bson(bson_data)
    
    # Extract state and action
    state, velocity, effort, action = extract_state_and_action_from_bson(bson_data)
    
    # Get episode index from path
    ep_idx = ep_path.split("/")[-1]
    
    # Get task prompt
    task_prompt, task_len = extract_task_prompt(additional_file, ep_idx, default_prompt)
    
    return imgs_per_cam, state, velocity, effort, action, task_prompt, task_len


def populate_dataset(
    dataset: LeRobotDataset,
    ep_dirs,
    episodes: list[int] | None = None,
    default_prompt: str = "do something",
    pad_end_len: int = 0,
) -> LeRobotDataset:
    """Populate the dataset with episode data."""
    if episodes is None:
        episodes = list(range(len(ep_dirs)))
    
    for ep_idx in tqdm.tqdm(episodes):
        ep_path = ep_dirs[ep_idx]
        
        # Load episode data
        imgs_per_cam, state, velocity, effort, action, task_prompt, task_len = load_bson_episode_data(
            ep_path, default_prompt=default_prompt
        )
        num_frames = state.shape[0]
        
        for i in range(num_frames):
            current_frame_data = {
                "observation.state": state[i],
                "observation.velocity": velocity[i],
                "observation.effort": effort[i],
                "action": action[i],
                "task": task_prompt[i],
                "task_len": np.array([task_len[i]], dtype=np.int32),
            }
            
            for camera, img_array in imgs_per_cam.items():
                current_frame_data[f"observation.images.{camera}"] = img_array[i]
            
            dataset.add_frame(current_frame_data)

            # Pad frames if it's the end of a task segment and pad_end_len > 0
            if task_len[i] == 1 and pad_end_len > 0:
                padding_frame_data = current_frame_data.copy()
                for _ in range(pad_end_len):
                    dataset.add_frame(padding_frame_data)

        dataset.save_episode()
    return dataset


def convert_bson_to_lerobot(
    bson_dir: Path,
    repo_id: str,
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    default_prompt: str = "do something",
    pad_end_len: int = 0,
):
    """Convert bson data to lerobot format.
    
    Args:
        bson_dir: Directory containing bson files
        repo_id: Repository ID for the lerobot dataset
        episodes: List of episode indices to convert (default: all episodes)
        push_to_hub: Whether to push the dataset to the hub
        mode: Whether to save images as videos or individual images
        dataset_config: Configuration for the dataset
        default_prompt: Default prompt to use when no task description is available
        pad_end_len: Number of frames to pad at the end of each task. Padded frames
                     will have the same content as the task's last frame, with task_len=1.
    
    Returns:
        The created lerobot dataset
    """
    # Find all bson directories
    ep_dirs = find_bson_dirs(bson_dir)
    print(f"Found {len(ep_dirs)} episode directories")
    
    if not ep_dirs:
        raise ValueError(f"No episode directories found in {bson_dir}")
    
    additional_file = Path(ep_dirs[0]) / ".." / "additional.json"
    # Check additional data if it exists
    if additional_file.exists():
        with open(additional_file, 'r') as f:
            additional_data = json.load(f)
        for key in additional_data.keys():
            split_frame_indices = additional_data[key]["marked_frames"]
            task_prompts = additional_data[key]["task_prompts"]
            if len(split_frame_indices) != len(task_prompts):
                print(f"Warning: Mismatch between number of splitting frames and task prompts for ep {key}: {len(split_frame_indices)} vs {len(task_prompts)}")
            if not all(task_prompts):
                print(f"Warning: Missing task_promts for ep {key}: {task_prompts}")
    else:
        print(f"Note: Additional data file not found at {additional_file}")
        print(f"Will use default task prompt for all frames: '{default_prompt}'")
    
    # Create empty dataset
    dataset = create_empty_dataset(
        repo_id=repo_id,
        mode=mode,
        has_velocity=False,
        has_effort=False,
        dataset_config=dataset_config,
    )
    
    # Populate dataset
    dataset = populate_dataset(
        dataset,
        ep_dirs,
        episodes,
        default_prompt=default_prompt,
        pad_end_len=pad_end_len
    )
    
    # Push to hub if requested
    if push_to_hub:
        dataset.push_to_hub()
    
    print(f"Converted {len(ep_dirs)} episodes to lerobot format")
    return dataset


if __name__ == "__main__":
    tyro.cli(convert_bson_to_lerobot)
