# scripts/prepare_amp_dataset.py
import sys
import os
sys.path.append(os.getcwd())

from pathlib import Path
import numpy as np
import joblib
from amp_rsl_rl.utils import download_amp_dataset_from_hf

def prepare_tiv2_amp_data():
    """准备TiV2的AMP训练数据"""
    
    # 方法1: 从现有的重定向数据转换
    print("Converting existing motion data to AMP format...")
    
    # 定义关节映射
    tiv2_joint_names = [
        'L_HIP_Y', 'L_HIP_R', 'L_HIP_P', 
        'L_KNEE_P', 'L_ANKLE_P', 'L_ANKLE_R',
        'R_HIP_Y', 'R_HIP_R', 'R_HIP_P',
        'R_KNEE_P', 'R_ANKLE_P', 'R_ANKLE_R'
    ]
    
    # 转换所有motion文件
    motion_dir = Path("/home/dy/dy/code/unitree_ti/data/ti512/v1/singles")
    output_dir = Path("/home/dy/dy/code/unitree_ti/amp_datasets/npy")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pkl_file in motion_dir.glob("*.pkl"):
        motion_data = joblib.load(pkl_file)
        
        # 转换为AMP格式
        for motion_name, data in motion_data.items():
            amp_data = {
                "joints_list": tiv2_joint_names,
                "joint_positions": list(data["dof"][:, :12]),  # 只取下半身
                "root_position": list(data["root_trans_offset"]),
                "root_quaternion": list(data["root_rot"]),  # xyzw格式
                "fps": data.get("fps", 30.0)
            }
            
            # 保存
            output_file = output_dir / f"{motion_name}.npy"
            np.save(output_file, amp_data)
            print(f"Saved: {output_file}")
    
    # 方法2: 从Hugging Face下载（如果有的话）
    # download_amp_dataset_from_hf(
    #     destination_dir=output_dir,
    #     robot_folder="tiv2",
    #     files=["walk.npy", "run.npy", "stand.npy"],
    #     repo_id="your-repo/amp-dataset"
    # )
    
    print("Dataset preparation complete!")
    return list(output_dir.glob("*.npy"))

if __name__ == "__main__":
    prepare_tiv2_amp_data()