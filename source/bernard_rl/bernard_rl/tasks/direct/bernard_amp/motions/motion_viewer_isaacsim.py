# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on spawning and interacting with an articulation."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import time
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from bernard import BERNARD_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

try:
    from .motion_loader import MotionLoader
except ImportError:
    from motion_loader import MotionLoader


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Articulation
    cfg = BERNARD_CFG.replace(prim_path="/World/envs/Robot")
    robot = Articulation(cfg=cfg)  # type: ignore[assignment]

    # Markers
    cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/bodyMarkers",
        markers={
            "body": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.08, 0.08, 0.08),
            ),
            "l_arm": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.08, 0.08, 0.08),
            ),
            "l_forearm": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.08, 0.08, 0.08),
            ),
            "l_foot": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.08, 0.08, 0.08),
            ),
            "r_arm": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.08, 0.08, 0.08),
            ),
            "r_forearm": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.08, 0.08, 0.08),
            ),
            "r_foot": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.08, 0.08, 0.08),
            ),
        },
    )
    marker = VisualizationMarkers(cfg)

    # return the scene information
    scene_entities = {"robot": robot, "marker": marker}
    return scene_entities, [[0.0, 0.0, 0.0]]


def run_simulator(
    sim: sim_utils.SimulationContext,
    entities: dict[str, Articulation],
    origins: torch.Tensor,
    motion_loader: MotionLoader,
) -> None:
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["robot"]
    marker: VisualizationMarkers = entities["marker"]
    # Read motion data from the motion loader
    dof_pos = motion_loader.dof_positions
    body_pos = motion_loader.body_positions
    body_rot = motion_loader.body_rotations
    active_dof_names = [
        "l_hip_joint",
        "r_hip_joint",
        "l_arm_joint",
        "r_arm_joint",
        "l_knee_joint",
        "r_knee_joint",
    ]
    motion_dof_idx = motion_loader.get_dof_index(active_dof_names)
    motion_root_idx = motion_loader.get_body_index(["body"])
    motion_body_idx = motion_loader.get_body_index(
        ["body", "l_arm", "l_forearm", "l_foot", "r_arm", "r_forearm", "r_foot"]
    )
    asset_active_dof_idx = [
        robot.data.joint_names.index(name) for name in active_dof_names
    ]
    asset_l_foot_dof_idx = [robot.data.joint_names.index("l_foot_joint")]
    asset_r_foot_dof_idx = [robot.data.joint_names.index("r_foot_joint")]
    motion_l_leg_dof_idx = motion_loader.get_dof_index(
        [name for name in active_dof_names if "l_" in name]
    )
    motion_r_leg_dof_idx = motion_loader.get_dof_index(
        [name for name in active_dof_names if "r_" in name]
    )
    l_leg_axis = torch.tensor([0.0, 1.0, -1.0], device=motion_loader.device)
    r_leg_axis = torch.tensor([0.0, -1.0, 1.0], device=motion_loader.device)
    # body_quat = motion_loader.body_rotations
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    frame = 0
    t = time.time()
    time_delta = 0.0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if frame % motion_loader.num_frames == 0:
            # reset counter
            frame = 0
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = body_pos[frame, motion_root_idx]
        root_state[:, 3:7] = body_rot[frame, motion_root_idx]
        root_state[:, 7:] = torch.zeros_like(root_state[:, 7:])
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        # set joint positions with some noise
        joint_pos, joint_vel = (
            robot.data.default_joint_pos.clone(),
            robot.data.default_joint_vel.clone(),
        )
        joint_pos[:, asset_active_dof_idx] = dof_pos[frame, motion_dof_idx]
        joint_pos[:, asset_l_foot_dof_idx] = torch.sum(
            dof_pos[frame, motion_l_leg_dof_idx] * l_leg_axis
        ).view(-1, 1)
        joint_pos[:, asset_r_foot_dof_idx] = torch.sum(
            dof_pos[frame, motion_r_leg_dof_idx] * r_leg_axis
        ).view(-1, 1)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        # Set markers
        marker.visualize(translations=body_pos[frame, motion_body_idx], orientations=body_rot[frame, motion_body_idx])
        # clear internal buffers
        robot.reset()
        # -- write data to sim
        robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        frame += 1
        # Wait to keep animation speed
        time_delta = time.time() - t
        if time_delta < motion_loader.dt:
            time.sleep(motion_loader.dt - time_delta)
        t = time.time()
        # Update buffers
        robot.update(sim_dt)


def main():
    """Main function."""
    # Motion data
    motion_loader = MotionLoader(
        device=torch.device(args_cli.device),
        motion_file="source/bernard_rl/bernard_rl/tasks/direct/bernard_amp/motions/bernard_walk.npz",
    )
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=motion_loader.dt)
    sim: SimulationContext = SimulationContext(sim_cfg)  # type: ignore[assignment]
    # Set main camera
    sim.set_camera_view((1.5, 0.0, 4.0), (0.3, 0.3, 1.0))
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins, motion_loader)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
