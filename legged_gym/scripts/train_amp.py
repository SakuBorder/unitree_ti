# scripts/train_tiv2_amp.py
import sys
import os
sys.path.append(os.getcwd())
sys.path.insert(0, '/home/dy/dy/code/unitree_ti/rsl_rl')

import isaacgym
from legged_gym.envs import *  # 这会自动执行__init__.py中的注册
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict  # 关键导入
from amp_rsl_rl.runners import AMPOnPolicyRunner

def train_amp(args):
    # 强制设置任务名确保使用AMP任务
    args.task = "tiv2_amp"
    print(f"=== Training Task: {args.task} ===")
    print(f"=== Using Device: {args.device} ===")
    
    # 验证任务注册
    available_tasks = list(task_registry.task_classes.keys())
    print(f"=== Available tasks: {available_tasks} ===")
    if args.task not in available_tasks:
        raise ValueError(f"Task {args.task} not registered")
    
    # 1. 创建环境（已经注册过了，直接使用）
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    print(f"=== Created environment type: {type(env)} ===")
    
    # 验证这是AMP环境
    if not hasattr(env, 'compute_amp_observations'):
        raise RuntimeError("Environment must have compute_amp_observations method for AMP training")
    
    print("=== Environment Observation Debug ===")
    try:
        # 调试环境观测接口
        obs_result = env.get_observations()
        if isinstance(obs_result, tuple):
            obs, extras = obs_result
            print(f"Actor obs shape: {obs.shape}")
            if "observations" in extras and "critic" in extras["observations"]:
                critic_obs = extras["observations"]["critic"]
                print(f"Critic obs shape (from extras): {critic_obs.shape}")
            if "observations" in extras and "amp" in extras["observations"]:
                amp_obs = extras["observations"]["amp"]
                print(f"AMP obs shape (from extras): {amp_obs.shape}")
        else:
            obs = obs_result
            print(f"Actor obs shape (single return): {obs.shape}")
        
        # 检查privileged observations
        if hasattr(env, 'get_privileged_observations'):
            priv = env.get_privileged_observations()
            print(f"Privileged obs shape: {priv.shape if priv is not None else 'None'}")
        else:
            print("No get_privileged_observations method")
            
        # 检查AMP observations
        if hasattr(env, 'compute_amp_observations'):
            amp_obs = env.compute_amp_observations()
            print(f"AMP obs shape: {amp_obs.shape}")
        else:
            print("No compute_amp_observations method")
            
    except Exception as e:
        print(f"Error during observation debug: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. 创建训练配置
    _, train_cfg = task_registry.get_cfgs(name=args.task)
    
    print("=== Training Configuration Debug ===")
    print(f"Runner class: {getattr(train_cfg.runner, 'runner_class_name', 'Not found')}")
    print(f"Policy class: {getattr(train_cfg.runner, 'policy_class_name', 'Not found')}")
    print(f"Algorithm class: {getattr(train_cfg.runner, 'algorithm_class_name', 'Not found')}")
    print(f"Num steps per env: {getattr(train_cfg.runner, 'num_steps_per_env', 'Not found')}")
    print(f"Max iterations: {getattr(train_cfg.runner, 'max_iterations', 'Not found')}")
    
    # 3. 根据训练配置选择合适的runner
    runner_class_name = getattr(train_cfg.runner, 'runner_class_name', 'AMPOnPolicyRunner')
    print(f"=== Config runner class: {runner_class_name} ===")
    
    if runner_class_name == 'AMPOnPolicyRunner':
        # 直接实例化我们的自定义AMPOnPolicyRunner
        print("=== Using Custom AMPOnPolicyRunner ===")
        
        # 确保log目录存在
        log_dir = f"./logs/{args.task}"
        os.makedirs(log_dir, exist_ok=True)
        print(f"=== Log directory: {log_dir} ===")
        
        # 关键修复：将配置对象转换为字典
        train_cfg_dict = class_to_dict(train_cfg)
        print(f"=== Converted config to dict, keys: {list(train_cfg_dict.keys())} ===")
        
        ppo_runner = AMPOnPolicyRunner(
            env=env,
            train_cfg=train_cfg_dict,  # 传入字典而不是对象
            log_dir=log_dir,
            device=args.device if hasattr(args, 'device') else 'cuda:0'
        )
    else:
        # 使用原有的task_registry方法
        print(f"=== Using Task Registry Runner: {runner_class_name} ===")
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env, 
            name=args.task, 
            args=args, 
            train_cfg=train_cfg
        )
    
    print("=== Final Runner Debug ===")
    print(f"Runner type: {type(ppo_runner)}")
    print(f"Environment type: {type(env)}")
    print(f"Device: {ppo_runner.device}")
    
    # 4. 最终观测兼容性检查
    try:
        print("=== Final Compatibility Check ===")
        if hasattr(ppo_runner, '_unpack_env_observations'):
            obs, priv = ppo_runner._unpack_env_observations(env)
            print(f"Runner unpacked - Actor: {obs.shape}, Critic: {priv.shape if priv is not None else 'None'}")
        
        if hasattr(ppo_runner, '_build_amp_obs_from_obs'):
            # 从配置中获取AMP观测维度
            if isinstance(train_cfg_dict, dict):
                num_amp_obs = train_cfg_dict.get('amp', {}).get('num_amp_obs', 30)
            else:
                num_amp_obs = 30
            print(f"Expected AMP obs dim: {num_amp_obs}")
            amp_obs = ppo_runner._build_amp_obs_from_obs(obs, num_amp_obs)
            print(f"AMP obs built: {amp_obs.shape}")
            
    except Exception as e:
        print(f"Warning during compatibility check: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. 开始训练
    print("=== Starting Training ===")
    try:
        max_iterations = train_cfg.runner.max_iterations if hasattr(train_cfg, 'runner') else 1000
        print(f"Max iterations: {max_iterations}")
        
        ppo_runner.learn(
            num_learning_iterations=max_iterations, 
            init_at_random_ep_len=True
        )
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    args = get_args()
    
    # 强制设置任务名，不依赖命令行参数
    args.task = "tiv2_amp"
    
    # 确保设备配置
    if not hasattr(args, 'device'):
        args.device = 'cuda:0'
    
    train_amp(args)