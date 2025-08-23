# debug_registration.py
import sys
sys.path.insert(0, '/home/dy/dy/code/unitree_ti/rsl_rl')
from legged_gym.envs import *
from legged_gym.utils.task_registry import task_registry

print("Available tasks:")
for name, task_class in task_registry.task_classes.items():
    print(f"  {name}: {task_class}")
    
print(f"\nTrying to get tiv2_amp config:")
try:
    env_cfg, train_cfg = task_registry.get_cfgs("tiv2_amp")
    print(f"Runner class: {train_cfg.runner.runner_class_name}")
    print(f"Environment: {task_registry.task_classes['tiv2_amp']}")
except Exception as e:
    print(f"Error: {e}")