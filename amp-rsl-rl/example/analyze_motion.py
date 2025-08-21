import numpy as np
from pathlib import Path

# 路径改成你的文件路径
path = Path("/home/dy/dy/code/unitree_ti/amp-rsl-rl/amp_datasets/ergocub/ref_motion.npy")

# 加载 .npy 文件
data = np.load(path, allow_pickle=True)

print("=== Basic Info ===")
print("File:", path)
print("Type:", type(data))

# 判断是字典还是单纯数组
if isinstance(data.item(), dict):
    data = data.item()
    print("Keys:", list(data.keys()))

    # 遍历每个字段
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f"\n--- {k} ---")
            print(" shape:", v.shape, " dtype:", v.dtype)
            print(" min:", np.min(v), " max:", np.max(v), " mean:", np.mean(v))
            print(" first frame:", v[0] if v.ndim > 1 else v[:10])
        else:
            print(f"\n--- {k} --- (non-array)")
            print(v)

else:
    print("Array shape:", data.shape, " dtype:", data.dtype)
    print(" min:", np.min(data), " max:", np.max(data), " mean:", np.mean(data))
    print(" first row:", data[0])
