# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Import required libraries
from pathlib import Path
from amp_rsl_rl.utils import AMPLoader, download_amp_dataset_from_hf
import torch
import sys
import shutil

# =============================================
# CONFIGURATION - 选择测试模式
# =============================================
TEST_MODE = "PKL_ONLY"  # 可选: "NPY_ONLY" 或 "PKL_ONLY"

print(f"[INFO] Test mode: {TEST_MODE}")

# =============================================
# NPY数据集配置
# =============================================
if TEST_MODE == "NPY_ONLY":
    repo_id = "ami-iit/amp-dataset"
    robot_folder = "ergocub"
    files = [
        "ergocub_stand_still.npy",
        "ergocub_walk_left0.npy", 
        "ergocub_walk.npy",
        "ergocub_walk_right2.npy",
    ]
    
    save_dir = Path("./amp_datasets") / robot_folder
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] NPY datasets will be saved to: {save_dir.resolve()}")
    
    # 下载NPY数据集
    try:
        dataset_names = download_amp_dataset_from_hf(
            save_dir,
            robot_folder=robot_folder,
            files=files,
            repo_id=repo_id,
        )
        print(f"[INFO] Downloaded datasets: {dataset_names}")
    except Exception as e:
        print(f"[ERROR] Failed to download NPY datasets: {e}")
        sys.exit(1)
    
    test_dir = save_dir
    test_datasets = dataset_names
    test_weights = [1.0] * len(dataset_names)

# =============================================
# PKL数据集配置  
# =============================================
elif TEST_MODE == "PKL_ONLY":
    # PKL文件路径
    pkl_file_path = Path("/home/dy/dy/code/unitree_ti/data/taihu/v1/singles/0-Male2Walking_c3d_B15 -  Walk turn around_poses.pkl")
    
    if not pkl_file_path.exists():
        print(f"[ERROR] PKL file not found: {pkl_file_path}")
        sys.exit(1)
    
    # 设置测试目录
    test_dir = Path("./amp_datasets/pkl_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制PKL文件
    dataset_name = "walk_turn_around"
    target_path = test_dir / f"{dataset_name}.pkl"
    shutil.copy2(pkl_file_path, target_path)
    print(f"[INFO] Copied PKL file to: {target_path}")
    
    # 检查PKL文件结构
    try:
        import joblib
        pkl_data = joblib.load(str(pkl_file_path))
        print(f"[INFO] PKL file structure:")
        print(f"       Type: {type(pkl_data)}")
        if isinstance(pkl_data, dict):
            first_key = list(pkl_data.keys())[0]
            first_motion = pkl_data[first_key]
            print(f"       Motion keys: {list(first_motion.keys())}")
            for key in ["root_trans_offset", "dof", "root_rot", "fps"]:
                if key in first_motion:
                    shape = getattr(first_motion[key], 'shape', 'N/A')
                    print(f"       {key}: {shape}")
    except Exception as e:
        print(f"[ERROR] Failed to inspect PKL file: {e}")
        sys.exit(1)
    
    test_datasets = [dataset_name]
    test_weights = [1.0]

else:
    print(f"[ERROR] Invalid TEST_MODE: {TEST_MODE}. Use 'NPY_ONLY' or 'PKL_ONLY'")
    sys.exit(1)

# =============================================
# 初始化AMPLoader
# =============================================
print(f"\n[INFO] Initializing AMPLoader for {TEST_MODE}...")
print(f"[INFO] Directory: {test_dir}")
print(f"[INFO] Datasets: {test_datasets}")

try:
    loader = AMPLoader(
        device="cuda" if torch.cuda.is_available() else "cpu",
        dataset_path_root=test_dir,
        dataset_names=test_datasets,
        dataset_weights=test_weights,
        simulation_dt=1/60.0,
        slow_down_factor=1,
        expected_joint_names=None,
    )
    print(f"[SUCCESS] AMPLoader initialized with {len(loader.motion_data)} datasets")
    
    for i, motion in enumerate(loader.motion_data):
        dataset_name = test_datasets[i]
        print(f"[INFO] {dataset_name}: {len(motion)} frames, {motion.joint_positions.shape[1]} joints")
        
except Exception as e:
    print(f"[ERROR] Failed to initialize AMPLoader: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================
# 功能测试
# =============================================
print(f"\n{'='*50}")
print("FUNCTIONALITY TESTS")
print("="*50)

motion = loader.motion_data[0]

# 测试1: 基本信息
print(f"[TEST 1] Dataset info:")
print(f"         Frames: {len(motion)}")
print(f"         Joint positions: {motion.joint_positions.shape}")
print(f"         Joint velocities: {motion.joint_velocities.shape}")
print(f"         Base quaternion: {motion.base_quat.shape}")

# 测试2: AMP观测
print(f"[TEST 2] AMP observations:")
try:
    sample_obs = motion.get_amp_dataset_obs(torch.tensor([0], device=motion.device))
    print(f"         Observation shape: {sample_obs.shape}")
    print(f"         First 10 values: {sample_obs[0, :10].tolist()}")
except Exception as e:
    print(f"[ERROR] {e}")

# 测试3: 重置状态
print(f"[TEST 3] Reset states:")
try:
    reset_states = loader.get_state_for_reset(3)
    print(f"         Components: {len(reset_states)}")
    for i, state in enumerate(reset_states):
        print(f"         State {i}: {state.shape}")
except Exception as e:
    print(f"[ERROR] {e}")

# 测试4: 批次生成
print(f"[TEST 4] Batch generation:")
try:
    batch_gen = loader.feed_forward_generator(num_mini_batch=2, mini_batch_size=4)
    for i, (obs, next_obs) in enumerate(batch_gen):
        print(f"         Batch {i}: {obs.shape} -> {next_obs.shape}")
        if i >= 1:
            break
except Exception as e:
    print(f"[ERROR] {e}")

# 测试5: 数据统计
print(f"[TEST 5] Data statistics:")
jp_min, jp_max = motion.joint_positions.min(), motion.joint_positions.max()
jv_min, jv_max = motion.joint_velocities.min(), motion.joint_velocities.max()
print(f"         Joint pos range: [{jp_min:.3f}, {jp_max:.3f}]")
print(f"         Joint vel range: [{jv_min:.3f}, {jv_max:.3f}]")
print(f"         Device: {motion.joint_positions.device}")
print(f"         Dtype: {motion.joint_positions.dtype}")

print(f"\n[SUCCESS] All tests completed for {TEST_MODE}!")
print(f"[INFO] Total frames: {sum(len(m) for m in loader.motion_data)}")
print(f"[INFO] Data saved in: {test_dir.resolve()}")