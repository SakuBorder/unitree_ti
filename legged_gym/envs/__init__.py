from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from legged_gym.envs.h1.h1_env import H1Robot
from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
from legged_gym.envs.h1_2.h1_2_env import H1_2Robot
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.envs.tienkung.tienkung_config import TienKungRoughCfg, TienKungRoughCfgPPO
from legged_gym.envs.tienkung.tienkung_env import TienKungRobot
from legged_gym.envs.tiv2.tiv2_config import TiV2RoughCfg, TiV2RoughCfgPPO
from legged_gym.envs.tiv2.tiv2_env import TiV2Robot
from legged_gym.envs.GR1.gr1_config import Gr1RoughCfg, Gr1RoughCfgPPO
from legged_gym.envs.GR1.gr1_env import Gr1Robot
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register( "h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())
task_registry.register( "h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO())
task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
task_registry.register( "tienkung", TienKungRobot, TienKungRoughCfg(), TienKungRoughCfgPPO())
task_registry.register( "tiv2", TiV2Robot, TiV2RoughCfg(), TiV2RoughCfgPPO())
task_registry.register( "gr1", Gr1Robot, Gr1RoughCfg(), Gr1RoughCfgPPO())