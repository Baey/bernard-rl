# Copyright (c) 2025, Błażej Szargut.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.envs.mdp import UniformVelocityCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    new_ranges: UniformVelocityCommandCfg.Ranges,
    num_steps: int = 100_000
):
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges

    # if env.common_step_counter > num_steps:
        # obtain the current command ranges
    x_min, x_max = ranges.lin_vel_x
    y_min, y_max = ranges.lin_vel_y
    heading_min, heading_max = ranges.heading
    # update the ranges
    x_min = new_ranges.lin_vel_x[0]
    x_max = new_ranges.lin_vel_x[1]
    y_min = new_ranges.lin_vel_y[0]
    y_max = new_ranges.lin_vel_y[1]
    heading_min = new_ranges.heading[0]
    heading_max = new_ranges.heading[1]
    # set the new ranges
    ranges.lin_vel_x = (x_min, x_max)
    ranges.lin_vel_y = (y_min, y_max)
    ranges.heading = (heading_min, heading_max)
    # update the command term configuration
    command_term.cfg.ranges = ranges
    env.command_manager.set_term_cfg("base_velocity", command_term.cfg)


def push_robot_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    velocity_range: dict[str, tuple[float, float]],
    interval_min: float = 5.0,
    alive_s_threshold: float = 12.0,
    velocity_step_size: float = 0.05,
    interval_step_size: float = 0.5
):
    """Curriculum based on the total environment timesteps

    This term is used to increase the difficulty of the task by pushing the robot with a random velocity
    when it is trained for a certain amount of time.

    Returns:
        The mean push level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    push_event = env.event_manager.get_term_cfg("push_robot")
    reward_term = env.reward_manager.get_term_cfg("alive")
    avg_alive_s = torch.mean(env.reward_manager._episode_sums["alive"][env_ids]) / reward_term.weight

    if env.common_step_counter % env.max_episode_length == 0:
        if avg_alive_s > alive_s_threshold:
            x_min = push_event.params['velocity_range']['x'][0]
            x_max = push_event.params['velocity_range']['x'][1]
            y_min = push_event.params['velocity_range']['y'][0]
            y_max = push_event.params['velocity_range']['y'][1]
            interval_lower = push_event.interval_range_s[0]
            interval_upper = push_event.interval_range_s[1]
            
            if x_min - velocity_step_size > velocity_range['x'][0]:
                x_min = max(x_min - velocity_step_size, velocity_range['x'][0])
            if x_max + velocity_step_size < velocity_range['x'][1]:
                x_max = min(x_max + velocity_step_size, velocity_range['x'][1])
            if y_min - velocity_step_size > velocity_range['y'][0]:
                y_min = max(y_min - velocity_step_size, velocity_range['y'][0])
            if y_max + velocity_step_size < velocity_range['y'][1]:
                y_max = min(y_max + velocity_step_size, velocity_range['y'][1])
            if interval_lower - interval_step_size > interval_min:
                interval_lower = max(interval_lower - interval_step_size, interval_min)
                interval_upper = interval_upper - interval_step_size

            push_event.params['velocity_range'] = {
                'x': (x_min, x_max),
                'y': (y_min, y_max)
            }
            push_event.interval_range_s = (interval_lower, interval_upper)
            env.event_manager.set_term_cfg("push_robot", push_event)


def reset_base_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    velocity_range: dict[str, tuple[float, float]],
    alive_s_threshold: float = 7.0,
    step_size: float = 0.05
):
    """Curriculum based on the total environment timesteps

    This term is used to increase the difficulty of the task by giving the robot a random velocity at
    the beginning of the episode.

    Returns:
        The mean velocity range for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    reset_event = env.event_manager.get_term_cfg("reset_base")
    reward_term = env.reward_manager.get_term_cfg("alive")
    avg_alive_s = torch.mean(env.reward_manager._episode_sums["alive"][env_ids]) / reward_term.weight

    if env.common_step_counter % env.max_episode_length == 0:
        if avg_alive_s > alive_s_threshold:
            x_min = reset_event.params['velocity_range']['x'][0]
            x_max = reset_event.params['velocity_range']['x'][1]
            y_min = reset_event.params['velocity_range']['y'][0]
            y_max = reset_event.params['velocity_range']['y'][1]
            z_min = reset_event.params['velocity_range']['z'][0]
            z_max = reset_event.params['velocity_range']['z'][1]

            if x_min - step_size > velocity_range['x'][0]:
                x_min = max(x_min - step_size, velocity_range['x'][0])
            if x_max + step_size < velocity_range['x'][1]:
                x_max = min(x_max + step_size, velocity_range['x'][1])
            if y_min - step_size > velocity_range['y'][0]:
                y_min = max(y_min - step_size, velocity_range['y'][0])
            if y_max + step_size < velocity_range['y'][1]:
                y_max = min(y_max + step_size, velocity_range['y'][1])
            if z_min - step_size > velocity_range['z'][0]:
                z_min = max(z_min - step_size, velocity_range['z'][0])
            if z_max + step_size < velocity_range['z'][1]:
                z_max = min(z_max + step_size, velocity_range['z'][1])
            reset_event.params['velocity_range'] = {
                'x': (x_min, x_max),
                'y': (y_min, y_max),
                'z': (z_min, z_max)
            }
            env.event_manager.set_term_cfg("reset_base", reset_event)


def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)