# Copyright (c) 2025, Błażej Szargut.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

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

from . import mdp

from bernard import BERNARD_CFG  # isort:skip


##
# Scene definition
##


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
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc, noise=Unoise(n_min=-0.01, n_max=0.01), scale=0.01
        )
        base_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel, noise=Unoise(n_min=-0.02, n_max=0.02), scale=0.1
        )
        base_orientation = ObsTerm(
            func=mdp.imu_orientation, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=".*_hip_.*|.*_arm_.*|.*_knee_.*"
                ),
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=".*_hip_.*|.*_arm_.*|.*_knee_.*"
                ),
            },
            scale=0.05
        )
        actions = ObsTerm(func=mdp.last_action)
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
            params={"period": 0.5}
        )

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
            func=mdp.imu_lin_acc, scale=0.01
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
            scale=0.05
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
        actions = ObsTerm(func=mdp.last_action)
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
            "dynamic_friction_range": (0.7, 0.99),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-0.1, 0.1),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (-0.7, 0.7),
            "torque_range": (-0.1, 0.1),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "yaw": (-3.14, 3.14),
                "z": (0.0, 0.1),
            },
            "velocity_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (-0.15, 0.15),
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
        interval_range_s=(8.0, 8.5),
        params={"velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # -- rewards
    feet_gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.11,
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
        weight=0.7,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
        }
    )
    alive = RewTerm(
        func=mdp.is_alive,
        weight=2.2,
    )
# 
    # -- penalties
    energy = RewTerm(func=mdp.energy, weight=-0.001)
    base_lin_vel_no_cmd = RewTerm(func=mdp.base_lin_vel_xy, weight=-0.56)
    base_height_l2 = RewTerm(func=mdp.base_height_l2, params={"target_height": 0.32}, weight=-13.5)
    base_pos_xyz = RewTerm(
        func=mdp.base_to_xy_l1,
        params={"target_pos": (0.0, 0.0)},
        weight=-3e-3
    )
    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.03)
    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # dof_torques_limits = RewTerm(func=mdp.applied_torque_limits, weight=-1.0e-6)
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-9)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*forearm.*|.*arm.*"), "threshold": 1.0},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.85)
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.85,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
        },
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-0.6,
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
        weight=-1.95,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_arm_.*"),
        }
    )
    joint_deviation_knees = RewTerm(
        func=mdp.joint_deviation_l1_tanh,
        weight=-1.8,
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
    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=-0.015,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_.*|.*_arm_.*|.*_knee_.*|.*_foot_.*"),
            "mirror_joints": [
                ["r_hip_joint", "l_hip_joint"],
                ["r_arm_joint", "l_arm_joint"],
                ["r_knee_joint", "l_knee_joint"],
                ["r_foot_joint", "l_foot_joint"],
            ]
        }
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.32,
        params={
            "std": 0.05,
            "tanh_mult": 1.5,
            "target_height": 0.14,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="body"),
            "threshold": 1.0,
        },
    )
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
            "minimum_height": 0.15,
        }
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # push_robot_curriculum = CurrTerm(
    #     func=mdp.push_robot_levels,
    #     params={
    #         "velocity_range": {
    #             "x": (-0.7, 0.7),
    #             "y": (-0.7, 0.7)
    #         },
    #         "alive_s_threshold": 12.0,
    #         "interval_min": 5.0,
    #     }
    # )
    # reset_base_curriculum = CurrTerm(
    #     func=mdp.reset_base_levels,
    #     params={
    #         "velocity_range": {
    #             "x": (-0.5, 0.5),
    #             "y": (-0.5, 0.5),
    #             "z": (-0.3, 0.3),
    #         },
    #         "alive_s_threshold": 7.0
    #     }
    # )
    modify_mirror_reward_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "joint_mirror",
            "weight": -0.001,
            "num_steps": 50_000,
        }
    )


##
# Environment configuration
##


@configclass
class BernardStandingEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: BernardSceneCfg = BernardSceneCfg(num_envs=128, env_spacing=2.5)
    # Basic settings
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
        self.decimation = 4
        self.episode_length_s = 15.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
