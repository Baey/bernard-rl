# Copyright (c) 2025, Błażej Szargut.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import numpy as np
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def feet_contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """"""
    # extract the used quantities (to enable type-hinting)
    feet_contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = torch.norm(feet_contact_sensor.data.net_forces_w_history, dim=-1)
    # compute the violation
    return net_contact_forces[:, 0, sensor_cfg.body_ids]


def gait_phase(env: ManagerBasedRLEnv, period: float, command_name: str | None=None) -> torch.Tensor:
    """Compute the gait phase for the agent."""
    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)

    if command_name is not None:
        scale = torch.clip(torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)/0.6, max=1.0, min=0.5)
        phase *= scale[:, None]
    return phase
