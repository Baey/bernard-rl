# Copyright (c) 2025, Błażej Szargut.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
import torch
from copy import deepcopy
import random

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg  # , RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.envs.mdp.curriculums import modify_env_param
from isaaclab.envs.mdp.curriculums import modify_env_param as mdp_modify_env_param

from . import mdp

from bernard import BERNARD_CFG  # isort:skip

CURRICULUM_STAGE_1_STEPS = 50_000
CURRICULUM_STAGE_2_STEPS = 150_000


##
# Scene definition
##

def override_velocity_command_range(env, env_ids, old_value, value, num_steps):
    # Override after num_steps
    if env.common_step_counter > num_steps:
        return value
    return mdp.modify_term_cfg.NO_CHANGE


@configclass
class BernardSceneCfg(InteractiveSceneCfg):
    """A generic configuration for the BERNARD bipedal robot scene."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # Make sure that the robot will face obstacles
    # robot
    robot: ArticulationCfg = BERNARD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=6,
        track_air_time=True,  # filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"]
    )

    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        debug_vis=True,
        update_period=0.01,
        # offset=ImuCfg.OffsetCfg(pos=(-0.07007606, -0.07839134, -0.23643290)),
    )

    # obstacle_usd_path = "/Isaac/Props/BasicShapes/cube.usd"

    # obstacle_1 = sim_utils.CuboidCfg(
    #         size=(0.3, 0.3, 0.3),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=False,
    #             kinematic_enabled=False,
    #         ),
    #     )

    # obstacle_1.func(
    #     prim_path="/World/Obstacle_1", cfg=obstacle_1, translation=(-0.2, 0.0, 2.0)
    # )

    # obstacle_2 = sim_utils.CuboidCfg(
    #     size=(0.3, 0.3, 0.3),
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         disable_gravity=False,
    #         kinematic_enabled=False,
    #     )
    # )

    # obstacle_2.func(
    #     prim_path="/World/Obstacle_2", cfg=obstacle_2, translation=(2.5, 0.5, 0.15)
    # )

    # obstacle_3 = sim_utils.CuboidCfg(
    #         size=(0.3, 0.3, 0.3),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=False,
    #             kinematic_enabled=False,
    #         ),
    #     )
    # obstacle_3.func(
    #     prim_path="/World/Obstacle_3", cfg=obstacle_3, translation=(3.0, -0.5, 0.15)
    # )

    
    obstacle_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_1",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.5, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),

            collision_props=sim_utils.CollisionPropertiesCfg(),

            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0, 0.0), metallic=0.2),
        ),
        
        init_state=ArticulationCfg.InitialStateCfg(
        pos=(1.0, 0.0, 0.075)),
        collision_group=-1
    )

    obstacle_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_2",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.5, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),

            collision_props=sim_utils.CollisionPropertiesCfg(),

            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            
        init_state=ArticulationCfg.InitialStateCfg(
        pos=(2.0, 0.0, 0.075)),
        collision_group=-1
    )

    obstacle_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_3",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.5, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),

            collision_props=sim_utils.CollisionPropertiesCfg(),

            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
            
        ),
        init_state=ArticulationCfg.InitialStateCfg(
        pos=(3.0, 0.0, 0.075)),
        collision_group=-1
    )

    # rigid_object_collection_cfg = RigidObjectCollectionCfg()
    # rigid_object_collection_cfg.rigid_objects = {"obstacle_1": obstacle_1, "obstacle_2": obstacle_2, "obstacle_3": obstacle_3}
    # rigid_object_collection = RigidObjectCollection(cfg=rigid_object_collection_cfg)
    # rigid_object_collections = {"obstacles": rigid_object_collection}


    


    # assign to a proper container field expected by InteractiveScene
    # assets = {
    #     "obstacle_1": obstacle_1,
    #     "obstacle_2": obstacle_2,
    #     "obstacle_3": obstacle_3,
    # }

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_pos = mdp.JointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=[".*_hip.*", ".*_arm.*", ".*_knee.*"],
    #     scale=0.25,
    #     # scale={
    #     #     "r_hip_joint": -0.52,
    #     #     "l_hip_joint": 0.52,
    #     #     "r_arm_joint": -1.22,
    #     #     "l_arm_joint": 1.22,
    #     #     "r_knee_joint": -1.75,
    #     #     "l_knee_joint": 1.75
    #     # },
    #     # offset={
    #     #     "r_hip_joint": 0.0,
    #     #     "l_hip_joint": 0.0,
    #     #     "r_arm_joint": 0.35,
    #     #     "l_arm_joint": -0.35,
    #     #     "r_knee_joint": 0.0,
    #     #     "l_knee_joint": 0.0
    #     # },
    #     use_default_offset=False
    # )
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*_hip.*", ".*_arm.*", ".*_knee.*"],
        scale=6.0
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.0, 0.0),
            lin_vel_y=(-0.0, 0.0),
            ang_vel_z=(-0.0, 0.0),
            heading=(-0.0, 0.0),
        )
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc, noise=Unoise(n_min=-0.01, n_max=0.01), scale=0.1, clip=(-30.0, 30.0)
        )
        base_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel, noise=Unoise(n_min=-0.15, n_max=0.15), scale=0.1
        )
        base_orientation = ObsTerm(
            func=mdp.imu_orientation
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            noise=Unoise(n_min=-0.08, n_max=0.08),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=".*_hip_.*|.*_arm_.*|.*_knee_.*"
                ),
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=".*_hip_.*|.*_arm_.*|.*_knee_.*"
                ),
            },
            scale=0.1
        )
        contact_forces = ObsTerm(
            func=mdp.feet_contact_forces,
            params={
                "threshold": 1.0,
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=".*foot"
                ),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=0.008
        )
        gait_phase = ObsTerm(
            func=mdp.gait_phase,
            params={
                "period": 0.5,
                "command_name": "base_velocity" 
            }
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        base_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc, scale=0.1, clip=(-30.0, 30.0)
        )
        base_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel, scale=0.1
        )
        base_orientation = ObsTerm(
            func=mdp.imu_orientation
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel
        )
        base_pos_z = ObsTerm(
            func=mdp.base_pos_z
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=".*_hip_.*|.*_arm_.*|.*_knee_.*|.*_foot_.*"
                ),
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=".*_hip_.*|.*_arm_.*|.*_knee_.*|.*_foot_.*"
                )
            },
            scale=0.1
        )
        joint_effort = ObsTerm(
            func=mdp.joint_effort,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=".*_hip_.*|.*_arm_.*|.*_knee_.*"
                )
            },
            scale=0.125
        )
        contact_forces = ObsTerm(
            func=mdp.feet_contact_forces,
            params={
                "threshold": 1.0,
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=".*foot"
                ),
            },
            scale=0.008
        )
        gait_phase = ObsTerm(
            func=mdp.gait_phase,
            params={"period": 0.5}
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        actions = ObsTerm(func=mdp.last_action)

    # observation groups
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.99),
            "dynamic_friction_range": (0.7, 0.95),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-0.15, 0.15),
            "operation": "add",
        },
    )
    move_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "com_range": {
                "x": (-0.08, 0.08),
                "y": (-0.06, 0.06),
                "z": (-0.05, 0.05),
            },
        },
    )
    move_rest_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!body$).*"),
            "com_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (-0.02, 0.02),
            },
        },
    )
    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "friction_distribution_params": (0.0, 0.015),
            "operation": "add"
        }
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "stiffness_distribution_params": (-30.0, 10.0),
            "damping_distribution_params": (-0.2, 0.1),
            "operation": "add"
        }
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (-0.7, 0.7),
            "torque_range": (-0.2, 0.2),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "yaw": (0.0, 0.0),
                "z": (0.0, 0.15),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.01, 0.01),
                "pitch": (-0.01, 0.01),
                "yaw": (-0.01, 0.01),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-0.3, 0.3),
            "velocity_range": (-0.1, 0.1),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 6.5),
        params={"velocity_range": {"x": (0.1, 0.1), "y": (0.4, 0.5)}},
    )

    
    # randomize_obstacle_height = EventTerm(
    #     mode="reset",
    #     params={
    #         "obstacle_cfgs": [
    #             SceneEntityCfg("Obstacle_1"),
    #             SceneEntityCfg("Obstacle_2"),
    #             SceneEntityCfg("Obstacle_3"),
    #         ],
    #         "x": [1.0, 2.0, 3.0],
    #         "y": [0.0, 0.0, 0.0],
    #         "z_range": [
    #             (0.05, 0.3),
    #             (0.05, 0.3),
    #             (0.05, 0.3),
    #         ],
    #     },
    #     func=lambda env, params: [
    #         env.scene.rigid_objects[params["obstacle_cfgs"][i].name].set_pose(
    #             position=(
    #                 params["x"][i],
    #                 params["y"][i],
    #                 np.random.uniform(*params["z_range"][i])
    #             ),
    #             orientation=(0.0, 0.0, 0.0, 1.0)  # upright
    #         )
    #         for i in range(len(params["obstacle_cfgs"]))
    #     ]
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.19)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # -- rewards
    feet_gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.35,
        params={
            "period": 0.5,
            "offset": [0.0, 0.5],
            "threshold": 0.7,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
        },
    )
    feet_on_ground_zero_cmd = RewTerm(
        func=mdp.feet_on_ground_zero_cmd,
        # weight=2.5,
        weight=0.72,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
            "command_name": "base_velocity",
        }
    )
    alive = RewTerm(
        func=mdp.is_alive,
        weight=2.2,
    )

    # -- penalties
    energy = RewTerm(func=mdp.energy, weight=-0.001)
    base_lin_vel_no_cmd = RewTerm(
        func=mdp.base_lin_vel_xy,
        params={"command_name": "base_velocity"},
        weight=-0.56
    )
    base_height_l2 = RewTerm(func=mdp.base_height_l2, params={"target_height": 0.32}, weight=-8.0)
    base_pos_xyz = RewTerm(
        func=mdp.base_to_xy_l1,
        params={"target_pos": (0.0, 0.0)},
        weight=-5e-3
    )
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-4.5e-3)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-1.2e-3)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*forearm.*|.*arm.*"), "threshold": 1.0},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.91)
    dof_torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-3.0e-3)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.95,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
        },
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-0.7,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
        }
    )
    joint_deviation_hips = RewTerm(
        func=mdp.joint_deviation_l1_tanh,
        weight=-0.55,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*hip_.*"),
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1_tanh,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_arm_.*"),
        }
    )
    joint_deviation_knees = RewTerm(
        func=mdp.joint_deviation_l1_tanh,
        weight=-1.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_knee_.*"),
        }
    )
    joint_deviation_foot = RewTerm(
        func=mdp.joint_deviation_l1_tanh,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_.*"),
        }
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "std": 0.04,
            "tanh_mult": 1.5,
            "target_height": 0.17,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="body"),
    #         "threshold": 1.0,
    #     },
    # )
    # bad_orientation = DoneTerm(
    #     func=mdp.bad_orientation,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="body"),
    #         "limit_angle": math.pi / 3,
    #     },
    # )
    root_height_below_minimum = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "minimum_height": 0.18,
        }
    )


def randomize_heights(env, env_ids, old_value, z_ranges):
    for i, val in enumerate(old_value):
        # Update the Z coordinate in-place
        val[2] = np.random.uniform(*z_ranges[i])
        # Apply the new pose to the actual environment object
        obstacle = env.scene.rigid_objects[i]  # assuming i corresponds to the right obstacle
        obstacle.set_pose(
            position=val,
            orientation=(0.0, 0.0, 0.0, 1.0)  # upright
        )


# def resample_obstacle_heights(env, env_ids, old_value, z_ranges):
#     """
#     Randomize obstacle heights at reset or during curriculum updates.

#     Args:
#         env: the IsaacLab environment
#         env_ids: list of environment indices
#         old_value: current obstacle positions (tensor or list)
#         z_ranges: list of (min, max) tuples for each obstacle

#     Returns:
#         Updated heights for each obstacle, same shape as old_value, or NO_CHANGE
#     """
#     # Randomize heights for each obstacle in env_ids
#     new_value = old_value.clone() if isinstance(old_value, torch.Tensor) else old_value.copy()

#     for i in range(len(env_ids)):
#         for j, (z_min, z_max) in enumerate(z_ranges):
#             new_z = np.random.uniform(z_min, z_max)
#             new_value[i, j, 2] = new_z  # assuming shape (env, obstacle, xyz)

#     return new_value

# def resample_obstacle_heights(env, env_ids, old_value, z_ranges):
#     """
#     Randomize the z positions (heights) of all obstacles directly via env.scene.

#     Args:
#         env: ManagerBasedRLEnv environment instance
#         env_ids: IDs of environments to update
#         old_value: unused (we modify in-place)
#         z_ranges: list of (min, max) tuples for each obstacle
#     """
#     # Loop through environments and obstacles
#     new_values = []
#     for env_id in env_ids:
#         for i, (z_min, z_max) in enumerate(z_ranges):
#             new_z = np.random.uniform(z_min, z_max)

#             # Retrieve the obstacle handle from the scene
#             obstacle_name = f"obstacle_{i+1}"
#             obstacle = getattr(env.scene, obstacle_name, None)
#             if obstacle is None:
#                 print("obstacle is None")
#                 continue

#             # Get current pose and update z
#             pose = obstacle.data.root_state_w[env_id].clone()
#             pose[2] = new_z  # modify z position
#             obstacle.write_root_state_to_sim(pose, env_ids=[env_id])
#             new_values.append(new_z)

#     # Return NO_CHANGE because we updated in-place
#     return new_values

# def resample_obstacle_heights(env, env_ids, old_value, z_ranges, every_n_steps=1):
#     """
#     Randomize the Z-position (height) of obstacles periodically or at reset.

#     This function is used by mdp.modify_env_param to modify
#     addresses such as scene.obstacle_1.init_state.pos.

#     Args:
#         env: The IsaacLab environment.
#         env_ids: The list of environment indices to modify.
#         old_value: The current tensor value of the addressed parameter
#                    (e.g. shape [num_envs, 3] for positions).
#         z_ranges: A list of (min, max) tuples specifying sampling ranges for Z.
#         every_n_steps: Only randomize every N global steps.

#     Returns:
#         torch.Tensor or modify_env_param.NO_CHANGE
#     """

#     # Only modify at specific intervals (to avoid jitter every frame)
#     if env.common_step_counter % every_n_steps != 0:
#         return modify_env_param.NO_CHANGE

#     # For each environment, assign a random Z height
#     z_min, z_max = z_ranges[0]
#     random_z = random.uniform(z_min, z_max)
#     new_value = (old_value[0], old_value[1], random_z) # modify Z component

#     return new_value

# def resample_obstacle_heights(env, env_ids, old_value, z_ranges, every_n_steps=5000):
#     """
#     Randomize the Z-position (height) of obstacles periodically or at reset.
#     Works on live RigidObject positions.

#     Args:
#         env: IsaacLab environment
#         env_ids: list of environment indices
#         old_value: placeholder (not used)
#         z_ranges: list of (min, max) tuples for each obstacle
#         every_n_steps: only randomize every N global steps

#     Returns:
#         torch.Tensor or mdp_modify_env_param.NO_CHANGE
#     """
#     if env.common_step_counter % every_n_steps != 0:
#         return mdp_modify_env_param.NO_CHANGE

#     # Iterate over all envs and obstacles
#     new_positions = []
#     for i, env_idx in enumerate(env_ids):
#         # Copy current positions
#         env_pos = []
#         for j, obstacle_name in enumerate(["obstacle_1", "obstacle_2", "obstacle_3"]):
#             obj = env.scene.rigid_objects[obstacle_name]
#             pose = obj.get_world_pose([env_idx])  # returns [pos, rot]
#             pos, rot = pose
#             z_min, z_max = z_ranges[j]
#             pos[0, 2] = random.uniform(z_min, z_max)  # change Z
#             # Apply new pose immediately
#             obj.set_world_pose(pos, rot, [env_idx])
#             env_pos.append(pos[0])  # only the first env in list
#         new_positions.append(env_pos)

#     return torch.tensor(new_positions, device="cpu")


def resample_obstacle_heights(env, env_ids, old_value, z_ranges, every_n_steps=5000):
    """
    Randomize the Z-position (height) of all rigid obstacles in the environment.

    Args:
        env: IsaacLab environment instance.
        env_ids: List of environment indices being reset/updated.
        old_value: Placeholder (not used).
        z_ranges: List of (min_z, max_z) tuples per obstacle or a single tuple for all.
        every_n_steps: Update frequency in environment steps.

    Returns:
        modify_env_param.NO_CHANGE (since we modify in-place)
    """
    # --- Update only at specified intervals ---
    if env.common_step_counter % every_n_steps != 0:
        return modify_env_param.NO_CHANGE

    # --- Access live RigidObjects in the scene ---
    rigid_objects = env.scene.rigid_objects

    # Normalize z_ranges if single range is given
    if len(z_ranges) == 1:
        z_ranges = z_ranges * len(rigid_objects)

    # --- For each obstacle, randomize its Z position ---
    for i, (name, obj) in enumerate(rigid_objects.items()):
        # Get current positions (tensor shape [num_envs, 3])
        pos = obj.data.root_link_pose_w

        # Randomize z for only selected env_ids
        z_min, z_max = z_ranges[i]
        new_z = torch.empty(len(env_ids), device=pos.device).uniform_(z_min, z_max)
        pos[env_ids, 2] = new_z

        # Optional: freeze rotation (keep same quat)
        # obj.root_quat_w[env_ids] = obj.root_quat_w[env_ids]

        # Push updated state to the simulation
        obj.write_root_pose_to_sim(root_pose=pos)

    return modify_env_param.NO_CHANGE



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # height_curriculum_1 = CurrTerm(
    #     func=mdp.modify_env_param,
    #     params={
    #         "address": "scene.rigid_objects.obstacle_1.cfg.init_state.pos",
    #         "modify_fn": resample_obstacle_heights,
    #         "modify_params": {"z_ranges": [(0.05, 0.3)]},
    #     },
    # )

    # height_curriculum_2 = CurrTerm(
    #     func=mdp.modify_env_param,
    #     params={
    #         "address": "scene.rigid_objects.obstacle_2.cfg.init_state.pos",
    #         "modify_fn": resample_obstacle_heights,
    #         "modify_params": {"z_ranges": [(0.05, 0.3)]},
    #     },
    # )

    # height_curriculum_3 = CurrTerm(
    #     func=mdp.modify_env_param,
    #     params={
    #         "address": "scene.rigid_objects.obstacle_3.cfg.init_state.pos",
    #         "modify_fn": resample_obstacle_heights,
    #         "modify_params": {"z_ranges": [(0.05, 0.3)]},
    #     },
    # )

    height_curriculum_1 = CurrTerm(
        func=mdp_modify_env_param,
        params={
            "address": "scene.rigid_objects",  # live objects container
            "modify_fn": resample_obstacle_heights,
            "modify_params": {"z_ranges": [(0.05, 0.3), (0.05, 0.3), (0.05, 0.3)], "every_n_steps": 1},
        },
    )

    push_robot_curriculum = CurrTerm(
        func=mdp.push_robot_levels,
        params={
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3)
            },
            "alive_s_threshold": 14.0,
            "interval_min": 7.0,
            "velocity_step_size": 0.1,
            "interval_step_size": 0.5
        }
    )
    reset_base_curriculum = CurrTerm(
        func=mdp.reset_base_levels,
        params={
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.3, 0.3),
            },
            "alive_s_threshold": 12.0,
            "step_size": 0.1
        }
    )
    velocity_command_x_curriculum_1 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": override_velocity_command_range,
            "modify_params": {
                "value": (-0.25, 0.25),
                "num_steps": CURRICULUM_STAGE_1_STEPS,
            }
        }
    )
    velocity_command_y_curriculum_1 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_y",
            "modify_fn": override_velocity_command_range,
            "modify_params": {
                "value": (-0.25, 0.25),
                "num_steps": CURRICULUM_STAGE_1_STEPS,
            }
        }
    )
    velocity_command_x_curriculum_2 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": override_velocity_command_range,
            "modify_params": {
                "value": (-0.45, 0.45),
                "num_steps": CURRICULUM_STAGE_2_STEPS,
            }
        }
    )
    velocity_command_y_curriculum_2 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_y",
            "modify_fn": override_velocity_command_range,
            "modify_params": {
                "value": (-0.45, 0.45),
                "num_steps": CURRICULUM_STAGE_2_STEPS,
            }
        }
    )
    velocity_command_z_curriculum_2 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.ang_vel_z",
            "modify_fn": override_velocity_command_range,
            "modify_params": {
                "value": (-0.35, 0.35),
                "num_steps": CURRICULUM_STAGE_2_STEPS,
            }
        }
    )
    velocity_command_heading_curriculum = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.heading",
            "modify_fn": override_velocity_command_range,
            "modify_params": {
                "value": (-math.pi, math.pi),
                "num_steps": CURRICULUM_STAGE_1_STEPS,
            }
        }
    )
    velocity_command_reward_schedule_1 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "track_lin_vel_xy_exp",
            "weight": 2.2,
            "num_steps": CURRICULUM_STAGE_1_STEPS,
        }
    )
    velocity_command_reward_schedule_2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "track_lin_vel_xy_exp",
            "weight": 3.0,
            "num_steps": CURRICULUM_STAGE_2_STEPS,
        }
    )
    velocity_z_command_reward_schedule_1 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "track_ang_vel_z_exp",
            "weight": 0.3,
            "num_steps": CURRICULUM_STAGE_1_STEPS,
        }
    )
    velocity_z_command_reward_schedule_1 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "track_ang_vel_z_exp",
            "weight": 0.5,
            "num_steps": CURRICULUM_STAGE_2_STEPS,
        }
    )
    base_pos_reward_schedule = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "base_pos_xyz",
            "weight": 0.0,
            "num_steps": CURRICULUM_STAGE_1_STEPS,
        }
    )
    joint_deviation_arms_reward_schedule = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "joint_deviation_arms",
            "weight": -0.8,
            "num_steps": CURRICULUM_STAGE_1_STEPS,
        }
    )
    joint_deviation_knees_reward_schedule = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "joint_deviation_knees",
            "weight": -0.4,
            "num_steps": CURRICULUM_STAGE_1_STEPS,
        }
    )
    joint_deviation_hips_reward_schedule = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "joint_deviation_hips",
            "weight": -0.1,
            "num_steps": CURRICULUM_STAGE_1_STEPS,
        }
    )

##
# Environment configuration
##


@configclass
class BernardFlatWithObstaclesEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: BernardSceneCfg = BernardSceneCfg(num_envs=128, env_spacing=2.5)
    # Basic settings
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 15.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class BernardFlatWithObstaclesEnvCfg_PLAY(BernardFlatWithObstaclesEnvCfg):
    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.commands.base_velocity.ranges.heading = (0, 0)
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
            "z": (-0.2, 0.2),
            "yaw": (-0.0, 0.0)
        }
        # self.events.push_robot = None
