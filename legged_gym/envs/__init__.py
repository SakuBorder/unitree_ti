from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.tiv2.tiv2_config import TiV2RoughCfg, TiV2RoughCfgPPO
from legged_gym.envs.tiv2.tiv2_env import TiV2Robot

from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.tiv2.ti_amp_env import TiV2AMPRobot
from legged_gym.envs.tiv2.ti_amp_config import TiV2AMPCfg, TiV2AMPCfgPPO
from legged_gym.envs.tiv2.ti_noamp_config import TiV2NoAMPCfg, TiV2NoAMPCfgPPO
task_registry.register( "tiv2", TiV2Robot, TiV2RoughCfg(), TiV2RoughCfgPPO())
task_registry.register("tiv2_amp", TiV2AMPRobot, TiV2AMPCfg(), TiV2AMPCfgPPO())
task_registry.register("tiv2_noamp", TiV2AMPRobot, TiV2NoAMPCfg(), TiV2NoAMPCfgPPO())
