# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Import required libraries
from pathlib import Path  # For path manipulation
from amp_rsl_rl.utils import (
    AMPLoader,
    download_amp_dataset_from_hf,
)  # Core AMP utilities
import torch  # PyTorch for tensor operations
import sys

# =============================================
# CONFIGURATION SECTION
# =============================================
# Define the dataset source and files to download
repo_id = "ami-iit/amp-dataset"  # Hugging Face repository ID
robot_folder = "ergocub"         # Subfolder containing robot-specific datasets

# List of motion dataset files to download
files = [
    "ergocub_stand_still.npy",   # Standing still motion
    "ergocub_walk_left0.npy",    # Walking left motion
    "ergocub_walk.npy",          # Straight walking motion
    "ergocub_walk_right2.npy",   # Walking right motion
]

# Persistent directory to save datasets (change if you like)
save_dir = Path("./amp_datasets") / robot_folder
save_dir.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Datasets will be saved to: {save_dir.resolve()}")

# =============================================
# DATASET DOWNLOAD AND LOADING
# =============================================
# Download datasets from Hugging Face Hub
# NOTE: Based on your earlier errors, this function expects the FIRST arg as a positional path,
# and accepts the remaining args as keywords (robot_folder, files, repo_id).
try:
    dataset_names = download_amp_dataset_from_hf(
        save_dir,                  # Where to save the files (positional)
        robot_folder=robot_folder, # Which robot dataset to use
        files=files,               # Which specific motion files to download
        repo_id=repo_id,           # Repository ID on Hugging Face Hub
    )
except ModuleNotFoundError as e:
    if "huggingface_hub" in str(e):
        print("[ERROR] Missing dependency 'huggingface_hub'. Please install it:\n  pip install huggingface_hub")
    else:
        print(f"[ERROR] Missing module: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Failed to download datasets: {e}")
    sys.exit(1)

# =============================================
# DATASET PROCESSING WITH AMPLoader
# =============================================
# Initialize the AMPLoader to process and manage the motion data
loader = AMPLoader(
    device="cpu",                       # Use CPU for processing (change to "cuda" for GPU)
    dataset_path_root=save_dir,         # Path to downloaded datasets
    dataset_names=dataset_names,        # Names of the loaded datasets (base names, no .npy)
    dataset_weights=[1.0] * len(dataset_names),  # Equal weights for all motions
    simulation_dt=1 / 60.0,             # Simulation timestep (60Hz)
    slow_down_factor=1,                 # Don't slow down the motions
    expected_joint_names=None,          # Use default joint ordering
)

# =============================================
# EXAMPLE USAGE
# =============================================
# Get the first motion sequence from the loader
if len(loader.motion_data) == 0:
    print("[ERROR] No motion data loaded.")
    sys.exit(1)

motion = loader.motion_data[0]

# Print basic information about the loaded motion
print("Loaded dataset with", len(motion), "frames.")

# Get and print a sample observation (first frame)
sample_obs = motion.get_amp_dataset_obs(torch.tensor([0]))  # Get frame 0
print("Sample AMP observation:", sample_obs)

# The motion data contains:
# - Joint positions and velocities
# - Base linear/angular velocities (local and world frames)
# - Base orientation (quaternion)

# Typical usage patterns:
# 1. For training: Use loader.feed_forward_generator() to get batches
# 2. For reset: Use loader.get_state_for_reset() to initialize robot states
# 3. For observation: Use motion.get_amp_dataset_obs() to get specific frames

print("\n[OK] Datasets are persisted at:", save_dir.resolve())
