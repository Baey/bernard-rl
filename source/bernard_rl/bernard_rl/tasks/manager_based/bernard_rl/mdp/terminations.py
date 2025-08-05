# Copyright (c) 2025, Błażej Szargut.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lying_on_back(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the robot is lying on its back.
    Args:
        env: The environment.
        threshold: The threshold for the difference between the gravity vector and the robot's up vector.
        asset_cfg: The configuration of the robot asset.
    Returns:
        A tensor indicating whether the robot is lying on its back.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity = asset.data.GRAVITY_VEC_W
    proj_gravity = asset.data.projected_gravity_b
    # print("up_vector", up_vector)
    # print("gravity", gravity)
    return torch.abs(torch.sum(gravity * proj_gravity, dim=1)) < threshold
