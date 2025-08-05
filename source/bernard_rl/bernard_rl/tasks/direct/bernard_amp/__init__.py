# Copyright (c) 2025, Błażej Szargut.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
AMP BERNARD locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# gym.register(
#     id="Isaac-Bernard-AMP-Walk-Slow-Direct-v0",
#     entry_point=f"{__name__}.bernard_amp_env:BernardAmpEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.bernard_amp_env_cfg:BernardAmpWalkSlowEnvCfg",
#         "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_walk_slow_amp_cfg.yaml",
#     },
# )

gym.register(
    id="Isaac-Bernard-AMP-Walk-Direct-v0",
    entry_point=f"{__name__}.bernard_amp_env:BernardAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bernard_amp_env_cfg:BernardAmpWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_walk_amp_cfg.yaml",
    },
)

# gym.register(
#     id="Isaac-Humanoid-AMP-Walk-Direct-v0",
#     entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpWalkEnvCfg",
#         "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_walk_amp_cfg.yaml",
#     },
# )