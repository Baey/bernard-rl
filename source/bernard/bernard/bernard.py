# Copyright (c) 2025, Błażej Szargut.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for BERNARD bipedal robot.

The following configurations are available:

* :obj:`BERNARD_CFG`: A bipedal robot with simple PD controller for the legs

Reference: https://github.com/UMich-BipedLab/Cassie_Model/blob/master/urdf/cassie.urdf
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

BERNARD_MODEL_DIR = os.path.join(os.path.dirname(__file__), "data", "bernard")
BERNARD_MODEL_PATH = os.path.join(BERNARD_MODEL_DIR, "bernard.usd")

BERNARD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=BERNARD_MODEL_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            "r_hip_joint": 0.0,
            "l_hip_joint": 0.0,
            "r_arm_joint": 0.0,
            "l_arm_joint": 0.0,
            "r_knee_joint": 0.0,
            "l_knee_joint": 0.0,
            "r_foot_joint": 0.0,
            "l_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "ak60_6": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_arm_.*", ".*_knee_.*"],
            effort_limit_sim=4.0,
            velocity_limit_sim=10.0,
            stiffness={
                ".*": 1.8,
            },
            damping={".*": 0.1},
        ),
        # "passive": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_foot_.*"],
        #     effort_limit_sim=0.0,
        #     velocity_limit_sim=0.0,
        #     stiffness={
        #         ".*": 0.0,
        #     },
        #     damping={".*": 0.05},
        # ),
    },
)
