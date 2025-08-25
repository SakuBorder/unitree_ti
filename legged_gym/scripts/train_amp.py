# scripts/train_tiv2_amp.py
import sys
import os
import time

sys.path.append(os.getcwd())
sys.path.insert(0, '/home/dy/dy/code/unitree_ti/rsl_rl')

import isaacgym  # noqa: F401
from legged_gym.envs import *  # noqa: F401  # 触发任务注册
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict


def _resolve_log_dir_from_cfg(train_cfg_dict: dict) -> str:
    """优先从配置里拿 experiment_name/run_name，缺省则给默认；最终返回 ./logs/<exp>/<run>。"""
    runner_cfg = train_cfg_dict.get("runner", {}) if isinstance(train_cfg_dict, dict) else {}
    exp = runner_cfg.get("experiment_name") or "tiv2_auto"
    run = runner_cfg.get("run_name") or time.strftime("run_%Y%m%d_%H%M%S")
    log_dir = os.path.join("./logs", exp, run)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _should_use_amp(env_cfg, train_cfg_dict) -> bool:
    """根据 env 与 runner/algorithm 名字，判断是否启用 AMP 流程。"""
    # env 开关（若未设置，默认为 True 以兼容 AMP 任务）
    amp_enabled_in_env = bool(getattr(getattr(env_cfg, "env", env_cfg), "enable_amp", True))
    # runner / algorithm 类名里含 AMP 视为 AMP 任务
    rcfg = train_cfg_dict.get("runner", {})
    runner_name = rcfg.get("runner_class_name", "")
    algo_name = rcfg.get("algorithm_class_name", "")
    return amp_enabled_in_env and (("AMP" in runner_name) or ("AMP" in algo_name))


def train_amp(args):
    # 使用命令行传入的 task（不再强制改成 tiv2_amp）
    task_name = args.task
    print(f"=== Training Task: {task_name} ===")
    print(f"=== Using Device: {args.device} ===")

    # 验证任务注册
    available_tasks = list(task_registry.task_classes.keys())
    print(f"=== Available tasks: {available_tasks} ===")
    if task_name not in available_tasks:
        raise ValueError(f"Task {task_name} not registered")

    # 1) 创建环境（基于当前 task）
    env, env_cfg = task_registry.make_env(name=task_name, args=args)
    print(f"=== Created environment type: {type(env)} ===")

    # 2) 观测调试（可选）
    print("=== Environment Observation Debug ===")
    try:
        obs_result = env.get_observations()
        if isinstance(obs_result, tuple):
            obs, extras = obs_result
            print(f"Actor obs shape: {obs.shape}")
            if isinstance(extras, dict) and "observations" in extras:
                if "critic" in extras["observations"]:
                    print(f"Critic obs shape (from extras): {extras['observations']['critic'].shape}")
                if "amp" in extras["observations"]:
                    print(f"AMP obs shape (from extras): {extras['observations']['amp'].shape}")
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

    # 3) 训练配置 -> dict
    _, train_cfg = task_registry.get_cfgs(name=task_name)
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

    # 4) 是否走 AMP
    use_amp = _should_use_amp(env_cfg, train_cfg_dict)
    print(f"=== AMP decision | env.enable_amp={getattr(getattr(env_cfg, 'env', env_cfg), 'enable_amp', True)} | "
          f"runner={rcfg.get('runner_class_name', '')} | algo={rcfg.get('algorithm_class_name', '')} "
          f"| use_amp={use_amp} ===")

    # 5) 实例化 Runner 并训练
    log_dir = _resolve_log_dir_from_cfg(train_cfg_dict)
    print(f"=== Log directory: {log_dir} ===")

    if use_amp:
        # AMP 路径
        try:
            from amp_rsl_rl.runners.amp_on_policy_runner import AMPOnPolicyRunner
        except Exception as e:
            raise RuntimeError(f"Failed to import AMPOnPolicyRunner: {e}")

        print("=== Using AMPOnPolicyRunner ===")
        runner = AMPOnPolicyRunner(
            env=env,
            train_cfg=train_cfg_dict,
            log_dir=log_dir,
            device=args.device if hasattr(args, 'device') else 'cuda:0'
        )
    else:
        # 纯 PPO 路径（无判别器、无 AMP 数据）
        # 即便配置里写了 AMP 的类名也强制改为 OnPolicyRunner/PPO
        rcfg['runner_class_name'] = 'OnPolicyRunner'
        rcfg['algorithm_class_name'] = 'PPO'
        train_cfg_dict['runner'] = rcfg

        # 尝试优先从 amp_rsl_rl 拿普通 Runner；若没有，请替换成你项目中的 OnPolicyRunner 路径
        try:
            from rsl_rl.rsl_rl.runners.him_on_policy_runner import HIMOnPolicyRunner
            print("=== Using OnPolicyRunner (NO AMP) ===")
            runner = HIMOnPolicyRunner(
                env=env,
                train_cfg=train_cfg_dict,
                log_dir=log_dir,
                device=args.device if hasattr(args, 'device') else 'cuda:0'
            )
        except Exception as e:
            # 退路：从 task_registry 用已有工厂创建（如果你的工程注册了 PPO runner）
            print(f"[Warn] Import OnPolicyRunner failed ({e}), fallback to task_registry.make_alg_runner")
            runner, _ = task_registry.make_alg_runner(
                env=env,
                name=task_name,
                args=args,
                train_cfg=train_cfg
            )

    print("=== Final Runner Debug ===")
    print(f"Runner type: {type(runner)}")
    print(f"Environment type: {type(env)}")
    if hasattr(runner, "device"):
        print(f"Device: {runner.device}")
    if hasattr(runner, "log_dir"):
        print(f"Final log_dir (runner): {runner.log_dir}")

    # 6) 最终兼容性检查（仅在 AMP 情况下做 AMP 维度检查）
    try:
        print("=== Final Compatibility Check ===")
        if hasattr(runner, '_unpack_env_observations'):
            _obs, _priv = runner._unpack_env_observations(env)
            print(f"Runner unpacked - Actor: {_obs.shape}, Critic: {_priv.shape if _priv is not None else 'None'}")

        if use_amp and hasattr(runner, '_build_amp_obs_from_obs'):
            num_amp_obs = train_cfg_dict.get('amp', {}).get('num_amp_obs', 30)
            print(f"Expected AMP obs dim: {num_amp_obs}")
            amp_obs_chk = runner._build_amp_obs_from_obs(_obs, num_amp_obs)
            print(f"AMP obs built: {amp_obs_chk.shape}")
    except Exception as e:
        print(f"Warning during compatibility check: {e}")

    # 7) 开始训练
    print("=== Starting Training ===")
    max_iterations = int(rcfg.get('max_iterations', 1000))
    print(f"Max iterations: {max_iterations}")
    runner.learn(
        num_learning_iterations=max_iterations,
        init_at_random_ep_len=True
    )


if __name__ == '__main__':
    args = get_args()
    if not hasattr(args, 'device'):
        args.device = 'cuda:0'
    # 不再强制 task 名；用命令行 --task 传入，比如：
    #   python scripts/train_tiv2_amp.py --task tiv2_noamp --headless
    train_amp(args)
