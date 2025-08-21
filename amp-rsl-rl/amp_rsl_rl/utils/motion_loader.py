# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import List, Union, Tuple, Generator, Optional
from dataclasses import dataclass
import joblib

import torch
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d


def download_amp_dataset_from_hf(
    destination_dir: Path,
    robot_folder: str,
    files: list,
    repo_id: str = "ami-iit/amp-dataset",
) -> list:
    """
    Downloads AMP dataset files from Hugging Face and saves them to `destination_dir`.
    Ensures real file copies (not symlinks or hard links).

    Args:
        destination_dir (Path): Local directory to save the files.
        robot_folder (str): Folder in the Hugging Face dataset repo to pull from.
        files (list): List of filenames to download.
        repo_id (str): Hugging Face repository ID. Default is "ami-iit/amp-dataset".

    Returns:
        List[str]: List of dataset names (without .npy extension).
    """
    from huggingface_hub import hf_hub_download

    destination_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = []

    for file in files:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{robot_folder}/{file}",
            repo_type="dataset",
            local_files_only=False,
        )
        local_copy = destination_dir / file
        # Deep copy to avoid symlinks
        with open(file_path, "rb") as src_file, open(local_copy, "wb") as dst_file:
            dst_file.write(src_file.read())
        
        # Remove extension for dataset name
        if file.endswith('.npy'):
            dataset_names.append(file.replace(".npy", ""))
        elif file.endswith('.pkl'):
            dataset_names.append(file.replace(".pkl", ""))
        else:
            dataset_names.append(file)

    return dataset_names


@dataclass
class MotionData:
    """
    Data class representing motion data for humanoid agents.

    This class stores joint positions and velocities, base velocities (both in local
    and mixed/world frames), and base orientation (as quaternion). It offers utilities
    for preparing data in AMP-compatible format, as well as environment reset states.

    Attributes:
        - joint_positions: shape (T, N)
        - joint_velocities: shape (T, N)
        - base_lin_velocities_mixed: linear velocity in world frame
        - base_ang_velocities_mixed: angular velocity in world frame
        - base_lin_velocities_local: linear velocity in local (body) frame
        - base_ang_velocities_local: angular velocity in local frame
        - base_quat: orientation quaternion as torch.Tensor in wxyz order

    Notes:
        - The quaternion is expected in the dataset as `xyzw` format (SciPy default),
          and it is converted internally to `wxyz` format to be compatible with IsaacLab conventions.
        - All data is converted to torch.Tensor on the specified device during initialization.
    """

    joint_positions: Union[torch.Tensor, np.ndarray]
    joint_velocities: Union[torch.Tensor, np.ndarray]
    base_lin_velocities_mixed: Union[torch.Tensor, np.ndarray]
    base_ang_velocities_mixed: Union[torch.Tensor, np.ndarray]
    base_lin_velocities_local: Union[torch.Tensor, np.ndarray]
    base_ang_velocities_local: Union[torch.Tensor, np.ndarray]
    base_quat: Union[Rotation, torch.Tensor]
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        # Convert numpy arrays (or SciPy Rotations) to torch tensors
        def to_tensor(x):
            return torch.tensor(x, device=self.device, dtype=torch.float32)

        if isinstance(self.joint_positions, np.ndarray):
            self.joint_positions = to_tensor(self.joint_positions)
        if isinstance(self.joint_velocities, np.ndarray):
            self.joint_velocities = to_tensor(self.joint_velocities)
        if isinstance(self.base_lin_velocities_mixed, np.ndarray):
            self.base_lin_velocities_mixed = to_tensor(self.base_lin_velocities_mixed)
        if isinstance(self.base_ang_velocities_mixed, np.ndarray):
            self.base_ang_velocities_mixed = to_tensor(self.base_ang_velocities_mixed)
        if isinstance(self.base_lin_velocities_local, np.ndarray):
            self.base_lin_velocities_local = to_tensor(self.base_lin_velocities_local)
        if isinstance(self.base_ang_velocities_local, np.ndarray):
            self.base_ang_velocities_local = to_tensor(self.base_ang_velocities_local)
        if isinstance(self.base_quat, Rotation):
            quat_xyzw = self.base_quat.as_quat()  # (T,4) xyzw
            # convert to wxyz
            self.base_quat = torch.tensor(
                quat_xyzw[:, [3, 0, 1, 2]],
                device=self.device,
                dtype=torch.float32,
            )
        elif isinstance(self.base_quat, np.ndarray):
            # Assume it's already in xyzw format from PKL
            if self.base_quat.shape[-1] == 4:
                # Convert xyzw to wxyz
                self.base_quat = torch.tensor(
                    self.base_quat[:, [3, 0, 1, 2]],
                    device=self.device,
                    dtype=torch.float32,
                )

    def __len__(self) -> int:
        return self.joint_positions.shape[0]

    def get_amp_dataset_obs(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Returns the AMP observation tensor for given indices.

        Args:
            indices: indices of samples to retrieve

        Returns:
            Concatenated observation tensor
        """
        return torch.cat(
            (
                self.joint_positions[indices],
                self.joint_velocities[indices],
                self.base_lin_velocities_local[indices],
                self.base_ang_velocities_local[indices],
            ),
            dim=1,
        )

    def get_state_for_reset(self, indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Returns the full state needed for environment reset.

        Args:
            indices: indices of samples to retrieve

        Returns:
            Tuple of (quat, joint_positions, joint_velocities, base_lin_velocities, base_ang_velocities)
        """
        return (
            self.base_quat[indices],
            self.joint_positions[indices],
            self.joint_velocities[indices],
            self.base_lin_velocities_local[indices],
            self.base_ang_velocities_local[indices],
        )

    def get_random_sample_for_reset(self, items: int = 1) -> Tuple[torch.Tensor, ...]:
        indices = torch.randint(0, len(self), (items,), device=self.device)
        return self.get_state_for_reset(indices)


class AMPLoader:
    """
    Extended loader and processor for humanoid motion capture datasets in AMP format.
    Now supports both NPY (original AMP format) and PKL (retargeted format) files.

    Responsibilities:
      - Loading .npy and .pkl files containing motion data
      - Building a unified joint ordering across all datasets
      - Resampling trajectories to match the simulator's timestep
      - Computing derived quantities (velocities, local-frame motion)
      - Returning torch-friendly MotionData instances

    Dataset format:
        NPY format - Each .npy contains a dict with keys:
          - "joints_list": List[str]
          - "joint_positions": List[np.ndarray]
          - "root_position": List[np.ndarray]
          - "root_quaternion": List[np.ndarray] (xyzw)
          - "fps": float (frames/sec)
        
        PKL format - Each .pkl contains a dict with motion name as key, value dict with:
          - "root_trans_offset": np.ndarray (T, 3)
          - "pose_aa": np.ndarray (T, num_bodies, 3) - axis-angle representation
          - "dof": np.ndarray (T, num_dof) - joint angles
          - "root_rot": np.ndarray (T, 4) - quaternion in xyzw
          - "smpl_joints": np.ndarray (T, J, 3) - optional, for reference
          - "fps": float

    Args:
        device: Target torch device ('cpu' or 'cuda')
        dataset_path_root: Directory containing the motion files
        dataset_names: List of dataset filenames (no extension)
        dataset_weights: List of sampling weights (for minibatch sampling)
        simulation_dt: Timestep used by the simulator
        slow_down_factor: Integer factor to slow down original data
        expected_joint_names: (Optional) override for joint ordering
    """

    def __init__(
        self,
        device: str,
        dataset_path_root: Path,
        dataset_names: List[str],
        dataset_weights: List[float],
        simulation_dt: float,
        slow_down_factor: int,
        expected_joint_names: Union[List[str], None] = None,
    ) -> None:
        self.device = device
        if isinstance(dataset_path_root, str):
            dataset_path_root = Path(dataset_path_root)

        # Detect file formats for each dataset
        self.dataset_formats = {}
        for name in dataset_names:
            npy_path = dataset_path_root / f"{name}.npy"
            pkl_path = dataset_path_root / f"{name}.pkl"
            
            if npy_path.exists():
                self.dataset_formats[name] = 'npy'
            elif pkl_path.exists():
                self.dataset_formats[name] = 'pkl'
            else:
                raise FileNotFoundError(f"Neither {npy_path} nor {pkl_path} found")

        # Build union of all joint names if not provided
        if expected_joint_names is None:
            joint_union: List[str] = []
            seen = set()
            for name in dataset_names:
                if self.dataset_formats[name] == 'npy':
                    p = dataset_path_root / f"{name}.npy"
                    info = np.load(str(p), allow_pickle=True).item()
                    for j in info["joints_list"]:
                        if j not in seen:
                            seen.add(j)
                            joint_union.append(j)
                # For PKL files, we'll use the provided expected_joint_names or skip joint name extraction
            expected_joint_names = joint_union if joint_union else []

        # Load and process each dataset into MotionData
        self.motion_data: List[MotionData] = []
        for dataset_name in dataset_names:
            dataset_path = dataset_path_root / f"{dataset_name}.{self.dataset_formats[dataset_name]}"
            
            if self.dataset_formats[dataset_name] == 'npy':
                md = self.load_npy_data(
                    dataset_path,
                    simulation_dt,
                    slow_down_factor,
                    expected_joint_names,
                )
            else:  # pkl
                md = self.load_pkl_data(
                    dataset_path,
                    simulation_dt,
                    slow_down_factor,
                    expected_joint_names,
                )
            
            if md is not None:
                self.motion_data.append(md)

        if not self.motion_data:
            raise ValueError("No valid motion data loaded")

        # Normalize dataset-level sampling weights
        weights = torch.tensor(dataset_weights[:len(self.motion_data)], dtype=torch.float32, device=self.device)
        self.dataset_weights = weights / weights.sum()

        # Precompute flat buffers for fast sampling
        obs_list, next_obs_list, reset_states = [], [], []
        for data, w in zip(self.motion_data, self.dataset_weights):
            T = len(data)
            idx = torch.arange(T, device=self.device)
            obs = data.get_amp_dataset_obs(idx)
            next_idx = torch.clamp(idx + 1, max=T - 1)
            next_obs = data.get_amp_dataset_obs(next_idx)

            obs_list.append(obs)
            next_obs_list.append(next_obs)

            quat, jp, jv, blv, bav = data.get_state_for_reset(idx)
            reset_states.append(torch.cat([quat, jp, jv, blv, bav], dim=1))

        self.all_obs = torch.cat(obs_list, dim=0)
        self.all_next_obs = torch.cat(next_obs_list, dim=0)
        self.all_states = torch.cat(reset_states, dim=0)

        # Build per-frame sampling weights: weight_i / length_i
        lengths = [len(d) for d in self.motion_data]
        per_frame = torch.cat(
            [
                torch.full((L,), w / L, device=self.device)
                for w, L in zip(self.dataset_weights, lengths)
            ]
        )
        self.per_frame_weights = per_frame / per_frame.sum()

    def _resample_data_Rn(
        self,
        data: Union[List[np.ndarray], np.ndarray],
        original_keyframes,
        target_keyframes,
    ) -> np.ndarray:
        """Resample data in R^n space using linear interpolation"""
        if isinstance(data, list):
            data = np.array(data)
        f = interp1d(original_keyframes, data, axis=0, fill_value="extrapolate")
        return f(target_keyframes)

    def _resample_data_SO3(
        self,
        raw_quaternions: Union[List[np.ndarray], np.ndarray],
        original_keyframes,
        target_keyframes,
    ) -> Rotation:
        """Resample rotations using SLERP"""
        if isinstance(raw_quaternions, list):
            raw_quaternions = np.array(raw_quaternions)
        # the quaternion is expected in the dataset as `xyzw` format (SciPy default)
        tmp = Rotation.from_quat(raw_quaternions)
        slerp = Slerp(original_keyframes, tmp)
        return slerp(target_keyframes)

    def _compute_ang_vel(
        self,
        data: List[Rotation],
        dt: float,
        local: bool = False,
    ) -> np.ndarray:
        """Compute angular velocities from rotation sequence"""
        R_prev = data[:-1]
        R_next = data[1:]

        if local:
            # Exp = R_i⁻¹ · R_{i+1}
            rel = R_prev.inv() * R_next
        else:
            # Exp = R_{i+1} · R_i⁻¹
            rel = R_next * R_prev.inv()

        # Log-map to rotation vectors and divide by Δt
        rotvec = rel.as_rotvec() / dt

        return np.vstack((rotvec, rotvec[-1]))

    def _compute_raw_derivative(self, data: np.ndarray, dt: float) -> np.ndarray:
        """Compute numerical derivative"""
        d = (data[1:] - data[:-1]) / dt
        return np.vstack([d, d[-1:]])

    def load_npy_data(
        self,
        dataset_path: Path,
        simulation_dt: float,
        slow_down_factor: int = 1,
        expected_joint_names: Union[List[str], None] = None,
    ) -> Optional[MotionData]:
        """
        Loads and processes one NPY motion dataset.

        Returns:
            MotionData instance or None if loading fails
        """
        try:
            data = np.load(str(dataset_path), allow_pickle=True).item()
        except Exception as e:
            print(f"Failed to load NPY file {dataset_path}: {e}")
            return None

        if not expected_joint_names:
            expected_joint_names = data.get("joints_list", [])

        dataset_joint_names = data.get("joints_list", [])

        # build index map for expected_joint_names
        idx_map: List[Union[int, None]] = []
        for j in expected_joint_names:
            if j in dataset_joint_names:
                idx_map.append(dataset_joint_names.index(j))
            else:
                idx_map.append(None)

        # reorder & fill joint positions
        jp_list: List[np.ndarray] = []
        for frame in data["joint_positions"]:
            arr = np.zeros((len(idx_map),), dtype=frame.dtype)
            for i, src_idx in enumerate(idx_map):
                if src_idx is not None:
                    arr[i] = frame[src_idx]
            jp_list.append(arr)

        dt = 1.0 / data["fps"] / float(slow_down_factor)
        T = len(jp_list)
        t_orig = np.linspace(0, T * dt, T)
        T_new = int(T * dt / simulation_dt)
        t_new = np.linspace(0, T * dt, T_new)

        resampled_joint_positions = self._resample_data_Rn(jp_list, t_orig, t_new)
        resampled_joint_velocities = self._compute_raw_derivative(
            resampled_joint_positions, simulation_dt
        )

        resampled_base_positions = self._resample_data_Rn(
            data["root_position"], t_orig, t_new
        )
        resampled_base_orientations = self._resample_data_SO3(
            data["root_quaternion"], t_orig, t_new
        )

        resampled_base_lin_vel_mixed = self._compute_raw_derivative(
            resampled_base_positions, simulation_dt
        )

        resampled_base_ang_vel_mixed = self._compute_ang_vel(
            resampled_base_orientations, simulation_dt, local=False
        )

        resampled_base_lin_vel_local = np.stack(
            [
                R.as_matrix().T @ v
                for R, v in zip(
                    resampled_base_orientations, resampled_base_lin_vel_mixed
                )
            ]
        )
        resampled_base_ang_vel_local = self._compute_ang_vel(
            resampled_base_orientations, simulation_dt, local=True
        )

        return MotionData(
            joint_positions=resampled_joint_positions,
            joint_velocities=resampled_joint_velocities,
            base_lin_velocities_mixed=resampled_base_lin_vel_mixed,
            base_ang_velocities_mixed=resampled_base_ang_vel_mixed,
            base_lin_velocities_local=resampled_base_lin_vel_local,
            base_ang_velocities_local=resampled_base_ang_vel_local,
            base_quat=resampled_base_orientations,
            device=self.device,
        )

    def load_pkl_data(
        self,
        dataset_path: Path,
        simulation_dt: float,
        slow_down_factor: int = 1,
        expected_joint_names: Union[List[str], None] = None,
    ) -> Optional[MotionData]:
        """
        Loads and processes one PKL motion dataset from retargeting.

        Returns:
            MotionData instance or None if loading fails
        """
        try:
            pkl_data = joblib.load(str(dataset_path))
        except Exception as e:
            print(f"Failed to load PKL file {dataset_path}: {e}")
            return None

        # PKL format: {motion_name: motion_data}
        if not isinstance(pkl_data, dict):
            print(f"Unexpected PKL format in {dataset_path}")
            return None

        motion_key = list(pkl_data.keys())[0]
        data = pkl_data[motion_key]

        # Extract required fields
        if "root_trans_offset" not in data or "dof" not in data:
            print(f"Missing required fields in PKL file {dataset_path}")
            return None

        root_trans = data["root_trans_offset"]  # (T, 3)
        dof_positions = data["dof"]  # (T, num_dof) or (T, num_dof, 1)
        root_quat_xyzw = data.get("root_rot", None)  # (T, 4) in xyzw format
        fps = data.get("fps", 30.0)

        # Handle different dof shapes
        if len(dof_positions.shape) == 3:
            dof_positions = dof_positions.squeeze(-1)

        # Convert to numpy if needed
        if isinstance(root_trans, torch.Tensor):
            root_trans = root_trans.numpy()
        if isinstance(dof_positions, torch.Tensor):
            dof_positions = dof_positions.numpy()
        if isinstance(root_quat_xyzw, torch.Tensor):
            root_quat_xyzw = root_quat_xyzw.numpy()

        # Resample to match simulation timestep
        dt = 1.0 / fps / float(slow_down_factor)
        T = len(dof_positions)
        t_orig = np.linspace(0, T * dt, T)
        T_new = int(T * dt / simulation_dt)
        if T_new == 0:
            print(f"Motion too short in {dataset_path}")
            return None
        t_new = np.linspace(0, T * dt, T_new)

        # Resample joint positions (DOF values)
        resampled_joint_positions = self._resample_data_Rn(dof_positions, t_orig, t_new)
        resampled_joint_velocities = self._compute_raw_derivative(
            resampled_joint_positions, simulation_dt
        )

        # Resample base positions
        resampled_base_positions = self._resample_data_Rn(root_trans, t_orig, t_new)
        resampled_base_lin_vel_mixed = self._compute_raw_derivative(
            resampled_base_positions, simulation_dt
        )

        # Handle rotations
        if root_quat_xyzw is not None:
            # PKL quaternions are in xyzw format, convert to Rotation for resampling
            resampled_base_orientations = self._resample_data_SO3(
                root_quat_xyzw, t_orig, t_new
            )
        else:
            # If no rotation data, use identity
            resampled_base_orientations = Rotation.identity(T_new)

        # Compute angular velocities
        resampled_base_ang_vel_mixed = self._compute_ang_vel(
            resampled_base_orientations, simulation_dt, local=False
        )
        resampled_base_ang_vel_local = self._compute_ang_vel(
            resampled_base_orientations, simulation_dt, local=True
        )

        # Compute local frame linear velocities
        resampled_base_lin_vel_local = np.stack(
            [
                R.as_matrix().T @ v
                for R, v in zip(
                    resampled_base_orientations, resampled_base_lin_vel_mixed
                )
            ]
        )

        return MotionData(
            joint_positions=resampled_joint_positions,
            joint_velocities=resampled_joint_velocities,
            base_lin_velocities_mixed=resampled_base_lin_vel_mixed,
            base_ang_velocities_mixed=resampled_base_ang_vel_mixed,
            base_lin_velocities_local=resampled_base_lin_vel_local,
            base_ang_velocities_local=resampled_base_ang_vel_local,
            base_quat=resampled_base_orientations,
            device=self.device,
        )

    def load_data(
        self,
        dataset_path: Path,
        simulation_dt: float,
        slow_down_factor: int = 1,
        expected_joint_names: Union[List[str], None] = None,
    ) -> Optional[MotionData]:
        """
        Loads and processes one motion dataset (auto-detects format).

        Returns:
            MotionData instance or None if loading fails
        """
        if dataset_path.suffix == '.pkl':
            return self.load_pkl_data(
                dataset_path, simulation_dt, slow_down_factor, expected_joint_names
            )
        elif dataset_path.suffix == '.npy':
            return self.load_npy_data(
                dataset_path, simulation_dt, slow_down_factor, expected_joint_names
            )
        else:
            print(f"Unsupported file format: {dataset_path.suffix}")
            return None

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Yields mini-batches of (state, next_state) pairs for training,
        sampled directly from precomputed buffers.

        Args:
            num_mini_batch: Number of mini-batches to yield
            mini_batch_size: Size of each mini-batch
        Yields:
            Tuple of (state, next_state) tensors
        """
        for _ in range(num_mini_batch):
            idx = torch.multinomial(
                self.per_frame_weights, mini_batch_size, replacement=True
            )
            yield self.all_obs[idx], self.all_next_obs[idx]

    def get_state_for_reset(self, number_of_samples: int) -> Tuple[torch.Tensor, ...]:
        """
        Randomly samples full states for environment resets,
        sampled directly from the precomputed state buffer.

        Args:
            number_of_samples: Number of samples to retrieve
        Returns:
            Tuple of (quat, joint_positions, joint_velocities, base_lin_velocities, base_ang_velocities)
        """
        idx = torch.multinomial(
            self.per_frame_weights, number_of_samples, replacement=True
        )
        full = self.all_states[idx]
        
        # Determine joint dimension from the first motion data
        if self.motion_data:
            joint_dim = self.motion_data[0].joint_positions.shape[1]
        else:
            # Fallback estimation
            joint_dim = (full.shape[1] - 10) // 2  # (total - 4quat - 3linvel - 3angvel) / 2

        # The dimensions of the full state are:
        #   - 4 (quat) + joint_dim (joint_positions) + joint_dim (joint_velocities)
        #   + 3 (base_lin_velocities) + 3 (base_ang_velocities)
        #   = 4 + joint_dim + joint_dim + 3 + 3
        dims = [4, joint_dim, joint_dim, 3, 3]
        return torch.split(full, dims, dim=1)