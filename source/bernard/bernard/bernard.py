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
from isaaclab.actuators import ActuatorNetLSTMCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

BERNARD_MODEL_DIR = os.path.join(os.path.dirname(__file__), "data", "BERNARD")
BERNARD_MODEL_PATH = os.path.join(BERNARD_MODEL_DIR, "BERNARD.usd")

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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=8
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            "r_hip_joint": 0.03,
            "l_hip_joint": -0.03,
            "r_arm_joint": -0.73,
            "l_arm_joint": 0.73,
            "r_knee_joint": -1.01,
            "l_knee_joint": 1.01,
            "r_foot_joint": -0.27,
            "l_foot_joint": -0.27,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "ak60_6": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_arm_.*", ".*_knee_.*"],
            effort_limit_sim=6.7,
            velocity_limit_sim=15.0,
            stiffness={
                ".*_hip_.*|.*_arm_.*|.*_knee_.*": 80.0,
            },
            damping={".*": 0.5},
        ),
        # "ak60_6": ActuatorNetLSTMCfg(
        #     joint_names_expr=[".*_hip_.*", ".*_arm_.*", ".*_knee_.*"],
        #     network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt",
        #     saturation_effort=10.0,
        #     effort_limit=8.0,
        #     velocity_limit=15.0,
        # ),
        # "passive": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_foot_.*"],
        #     effort_limit_sim=0.5,
        #     velocity_limit_sim=1.0,
        #     stiffness={
        #         ".*": 1.0,
        #     },
        #     damping={".*": 0.05},
        # ),
    },
)
