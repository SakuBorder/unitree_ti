# scripts/play_tiv2_amp.py
import os, sys, shutil, time
sys.path.append(os.getcwd())
sys.path.insert(0, '/home/dy/dy/code/unitree_ti/rsl_rl')

import isaacgym
import torch
from torch.utils.tensorboard import SummaryWriter

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
from amp_rsl_rl.runners import AMPOnPolicyRunner

def make_env_for_test(args):
    # 取配置并做轻量化设置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 8)
    if hasattr(env_cfg, "terrain"):
        env_cfg.terrain.num_rows = 5
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.curriculum = False
    if hasattr(env_cfg, "noise"):
        env_cfg.noise.add_noise = False
    if hasattr(env_cfg, "domain_rand"):
        # 为了可视化更稳定，回放时减弱随机化（按需打开）
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_kp = False
        env_cfg.domain_rand.randomize_kd = False
    env_cfg.env.test = True

    # ⚠️ 不要把命令区间改成 0；阶段A在环境里已冻结为常量了，这里只确保不重采样
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.resampling_time = 1e12  # 形同禁用重采样

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # （可选）热修补：某些实现会在 step 中重采样命令，这里直接屏蔽掉
    if hasattr(env, "resample_commands"):
        env.resample_commands = lambda *a, **k: None
    if hasattr(env, "update_command"):
        env.update_command = lambda *a, **k: None
    if hasattr(env, "cfg") and hasattr(env.cfg, "commands"):
        try:
            env.cfg.commands.resampling_time = 1e12
        except Exception:
            pass

    return env, train_cfg

def _resolve_log_root(train_cfg):
    # 允许用环境变量覆盖（绝对路径或相对路径都行）
    env_root = os.environ.get("AMP_LOG_ROOT", None)
    if env_root:
        return env_root

    runner = getattr(train_cfg, "runner", object)
    exp = getattr(runner, "experiment_name", None)
    run = getattr(runner, "run_name", None)

    # 旧版兜底：只有 task 目录
    if exp is None and run is None:
        return "./logs/tiv2_amp"

    # 新版默认：logs/<exp>/<run>（若没有 run，就只到 <exp>）
    if exp is None:
        exp = "tiv2_amp"
    base = os.path.join("./logs", exp)
    return os.path.join(base, run) if run else base


def load_policy_with_runner(env, train_cfg, device):
    train_cfg_dict = class_to_dict(train_cfg)
    runner = AMPOnPolicyRunner(env=env, train_cfg=train_cfg_dict, log_dir=None, device=device)

    # ① 期望目录：logs/<exp>/<run> 或环境变量指定
    log_root = _resolve_log_root(train_cfg)

    # ② 如果该目录没有 checkpoint，尝试：
    #    a) 上一级（logs/<exp>）
    #    b) 下一级（logs/<exp>/*/）
    #    c) 直接用环境变量 AMP_LOG_ROOT（若设置的是具体文件夹也能命中）
    candidates = []
    if os.path.isdir(log_root):
        candidates.append(log_root)

    parent = os.path.dirname(log_root)
    if os.path.isdir(parent):
        candidates.append(parent)

    if os.path.isdir(os.path.join(log_root, "tiv2_amp_phaseA_run")):
        candidates.append(os.path.join(log_root, "tiv2_amp_phaseA_run"))

    # 向下搜索一层 run 目录
    if os.path.isdir(parent):
        for name in os.listdir(parent):
            p = os.path.join(parent, name)
            if os.path.isdir(p):
                candidates.append(p)

    # 去重，保持顺序
    seen = set(); ordered = []
    for c in candidates:
        if c not in seen:
            ordered.append(c); seen.add(c)

    # 在候选目录中查找 model_*.pt
    ckpt_path = None
    for root in ordered:
        ckpts = [f for f in os.listdir(root) if f.startswith("model_") and f.endswith(".pt")]
        if ckpts:
            ckpts.sort(key=lambda n: int(n.split("_")[1].split(".")[0]))
            ckpt_path = os.path.join(root, ckpts[-1])
            log_root = root
            break

    if ckpt_path is None:
        raise FileNotFoundError(
            "No checkpoints found. Tried:\n  " + "\n  ".join(ordered) +
            "\nTip: export AMP_LOG_ROOT=/absolute/path/to/logdir or point to the run folder."
        )

    print(f"[PLAY] Loading checkpoint: {ckpt_path}")
    runner.load(ckpt_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    return policy, log_root


def play(args):
    # 任务名与训练环境注册名一致（你用的是 TiV2AMPRobot 对应的 "tiv2_amp"）
    args.task = "tiv2_noamp"

    env, train_cfg = make_env_for_test(args)
    device = env.device

    # 可选：重置一次
    if hasattr(env, "reset"):
        env.reset()

    # 取观测（兼容返回 (obs, extras) 的情况）
    got = env.get_observations()
    obs = got[0] if isinstance(got, tuple) else got

    policy, log_root = load_policy_with_runner(env, train_cfg, device)

    # TensorBoard
    log_dir = os.path.join('debug', 'gym')
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # ====== 目标命令：与阶段A训练时保持一致 ======
    # 环境已冻结命令为常量，这里取 cfg 中的常量，避免分布偏移
    if hasattr(env, "cfg") and hasattr(env.cfg, "commands"):
        TARGET_VX = getattr(env.cfg.commands, "phaseA_vx", 0.8)
        TARGET_VY = getattr(env.cfg.commands, "phaseA_vy", 0.0)
        TARGET_YAW = getattr(env.cfg.commands, "phaseA_yaw", 0.0)
    else:
        TARGET_VX, TARGET_VY, TARGET_YAW = 0.8, 0.0, 0.0

    steps = 10 * int(env.max_episode_length)
    for i in range(steps):
        with torch.no_grad():
            # 先把指令钉住，再算动作（双保险）
            if hasattr(env, "commands"):
                env.commands[:, 0] = TARGET_VX
                env.commands[:, 1] = TARGET_VY
                env.commands[:, 2] = TARGET_YAW

            actions = policy(obs.detach())

        # 再钉一次，确保进入 step 前命令没被别处改
        if hasattr(env, "commands"):
            env.commands[:, 0] = TARGET_VX
            env.commands[:, 1] = TARGET_VY
            env.commands[:, 2] = TARGET_YAW

        # if hasattr(env, "commands") and (i < 10 or i % 50 == 0):
        #     print(f"[{i:06d}] cmd before step:", env.commands[0].detach().cpu().tolist())

        # 你的 env.step 返回 7 元组
        obs, _, rews, dones, infos, _, _ = env.step(actions)

        # if hasattr(env, "commands") and (i < 10 or i % 50 == 0):
        #     print(f"[{i:06d}] cmd after  step:", env.commands[0].detach().cpu().tolist())

        # 保险：若 step 内被改动，step 后再钉回
        if hasattr(env, "commands"):
            env.commands[:, 0] = TARGET_VX
            env.commands[:, 1] = TARGET_VY
            env.commands[:, 2] = TARGET_YAW

        # 记录速度，看看是否在 +x 方向匀速前进
        if hasattr(env, "base_lin_vel"):
            vx = env.base_lin_vel[0, 0].item()
            vy = env.base_lin_vel[0, 1].item()
            writer.add_scalar('eval/base_lin_vel_x', vx, i)
            writer.add_scalar('eval/base_lin_vel_y', vy, i)

        # 你的可视化
        for j in range(min(6, actions.shape[1] // 2)):
            writer.add_scalars(
                f'fig_gym/dim_{j}_vs_dim_{j+6}',
                {f'dim_{j}': actions[0, j].item(), f'dim_{j+6}': actions[0, j+6].item()},
                i
            )

        if hasattr(env, "base_ang_vel"):
            writer.add_scalar('fig/base_ang_vel_x', env.base_ang_vel[0, 0].item(), i)
            writer.add_scalar('fig/base_ang_vel_y', env.base_ang_vel[0, 1].item(), i)

        if isinstance(infos, dict) and 'feet_contact_forces' in infos:
            fcf = infos['feet_contact_forces']  # [num_env, 2 feet, 3]
            writer.add_scalars('fig_gym/feet_contact_force_z',
                               {'left': fcf[0, 0, 2].item(), 'right': fcf[0, 1, 2].item()},
                               i)

        writer.add_histogram('action_histogram', actions[0].cpu().numpy(), i)

    writer.close()
    print(f"[PLAY] Done. TensorBoard logs at: {log_dir}")
    print(f"[PLAY] Loaded checkpoints from: {log_root}")

if __name__ == '__main__':
    args = get_args()
    # 小批次回放
    args.num_envs = 1
    play(args)
