# scripts/train_tiv2_amp.py
import sys
import os
import time

sys.path.append(os.getcwd())
sys.path.insert(0, '/home/dy/dy/code/unitree_ti/rsl_rl')

import isaacgym  # noqa: F401
from legged_gym.envs import *  # noqa: F401  # 触发注册
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
from amp_rsl_rl.runners import AMPOnPolicyRunner


def _resolve_log_dir_from_cfg(train_cfg_dict: dict) -> str:
    """优先从配置里拿 experiment_name/run_name，缺省则给默认；最终返回 ./logs/<exp>/<run>。"""
    runner_cfg = train_cfg_dict.get("runner", {}) if isinstance(train_cfg_dict, dict) else {}
    exp = runner_cfg.get("experiment_name") or "tiv2_amp"          # 默认到任务名
    run = runner_cfg.get("run_name") or time.strftime("run_%Y%m%d_%H%M%S")
    log_dir = os.path.join("./logs", exp, run)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def train_amp(args):
    # 强制使用 AMP 任务
    args.task = "tiv2_amp"
    print(f"=== Training Task: {args.task} ===")
    print(f"=== Using Device: {args.device} ===")

    # 验证任务注册
    available_tasks = list(task_registry.task_classes.keys())
    print(f"=== Available tasks: {available_tasks} ===")
    if args.task not in available_tasks:
        raise ValueError(f"Task {args.task} not registered")

    # 1) 创建环境
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    print(f"=== Created environment type: {type(env)} ===")

    # 基础校验
    if not hasattr(env, 'compute_amp_observations'):
        raise RuntimeError("Environment must have compute_amp_observations for AMP training")

    # 观测调试（可选）
    print("=== Environment Observation Debug ===")
    try:
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

        if hasattr(env, 'get_privileged_observations'):
            priv = env.get_privileged_observations()
            print(f"Privileged obs shape: {priv.shape if priv is not None else 'None'}")

        if hasattr(env, 'compute_amp_observations'):
            amp_obs = env.compute_amp_observations()
            print(f"AMP obs shape: {amp_obs.shape}")
    except Exception as e:
        print(f"Error during observation debug: {e}")

    # 2) 拿训练配置并转 dict（后续统一用 dict）
    _, train_cfg = task_registry.get_cfgs(name=args.task)
    train_cfg_dict = class_to_dict(train_cfg)

    print("=== Training Configuration Debug ===")
    rcfg = train_cfg_dict.get("runner", {})
    print(f"Runner class: {rcfg.get('runner_class_name', 'Not found')}")
    print(f"Policy class: {rcfg.get('policy_class_name', 'Not found')}")
    print(f"Algorithm class: {rcfg.get('algorithm_class_name', 'Not found')}")
    print(f"Num steps per env: {rcfg.get('num_steps_per_env', 'Not found')}")
    print(f"Max iterations: {rcfg.get('max_iterations', 'Not found')}")
    print(f"experiment_name: {rcfg.get('experiment_name', '(default)')}")
    print(f"run_name: {rcfg.get('run_name', '(timestamp)')}")

    # 3) 选择 Runner 并实例化（日志目录从配置解析）
    runner_class_name = rcfg.get('runner_class_name', 'AMPOnPolicyRunner')
    print(f"=== Config runner class: {runner_class_name} ===")

    if runner_class_name == 'AMPOnPolicyRunner':
        print("=== Using Custom AMPOnPolicyRunner ===")
        log_dir = _resolve_log_dir_from_cfg(train_cfg_dict)
        print(f"=== Log directory: {log_dir} ===")

        ppo_runner = AMPOnPolicyRunner(
            env=env,
            train_cfg=train_cfg_dict,
            log_dir=log_dir,  # 由配置解析出的路径
            device=args.device if hasattr(args, 'device') else 'cuda:0'
        )
    else:
        print(f"=== Using Task Registry Runner: {runner_class_name} ===")
        ppo_runner, _ = task_registry.make_alg_runner(
            env=env,
            name=args.task,
            args=args,
            train_cfg=train_cfg
        )

    print("=== Final Runner Debug ===")
    print(f"Runner type: {type(ppo_runner)}")
    print(f"Environment type: {type(env)}")
    print(f"Device: {ppo_runner.device}")
    if hasattr(ppo_runner, "log_dir"):
        print(f"Final log_dir (runner): {ppo_runner.log_dir}")

    # 4) 最终观测兼容性检查（可选）
    try:
        print("=== Final Compatibility Check ===")
        if hasattr(ppo_runner, '_unpack_env_observations'):
            obs, priv = ppo_runner._unpack_env_observations(env)
            print(f"Runner unpacked - Actor: {obs.shape}, Critic: {priv.shape if priv is not None else 'None'}")

        if hasattr(ppo_runner, '_build_amp_obs_from_obs'):
            num_amp_obs = train_cfg_dict.get('amp', {}).get('num_amp_obs', 30)
            print(f"Expected AMP obs dim: {num_amp_obs}")
            amp_obs = ppo_runner._build_amp_obs_from_obs(obs, num_amp_obs)
            print(f"AMP obs built: {amp_obs.shape}")
    except Exception as e:
        print(f"Warning during compatibility check: {e}")

    # 5) 开始训练
    print("=== Starting Training ===")
    try:
        max_iterations = rcfg.get('max_iterations', 1000)
        print(f"Max iterations: {max_iterations}")
        ppo_runner.learn(
            num_learning_iterations=int(max_iterations),
            init_at_random_ep_len=True
        )
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == '__main__':
    args = get_args()
    args.task = "tiv2_amp"  # 强制任务名
    if not hasattr(args, 'device'):
        args.device = 'cuda:0'
    train_amp(args)
