# scripts/train_tiv2_amp.py
import sys
import os
sys.path.append(os.getcwd())
sys.path.insert(0, '/home/dy/dy/code/unitree_ti/rsl_rl')

import isaacgym
from legged_gym.envs import *  # 这会自动执行__init__.py中的注册
from legged_gym.utils import get_args, task_registry
from amp_rsl_rl.runners import AMPOnPolicyRunner

def train_amp(args):
    # 1. 创建环境（已经注册过了，直接使用）
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    # 2. 创建训练配置
    _, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 3. 创建AMP runner（使用已有的方法）
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, 
        name=args.task, 
        args=args, 
        train_cfg=train_cfg
    )
    
    # 4. 开始训练
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations, 
        init_at_random_ep_len=True
    )

if __name__ == '__main__':
    args = get_args()
    args.task = "tiv2_amp"  # 使用已注册的任务名
    train_amp(args)