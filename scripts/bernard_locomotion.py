# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates an interactive demo with the H1 rough terrain environment.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/h1_locomotion.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import os

os.environ["ROS_DOMAIN_ID"] = "0" 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
import scripts.rsl_rl.cli_args as cli_args  # isort: skip


from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Interactive demo with the Bernard robot."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
import rclpy # type: ignore # noqa: E261
from rclpy.node import Node # type: ignore # noqa: E261
from sensor_msgs.msg import Imu, JointState # type: ignore # noqa: E261
from std_msgs.msg import Float32MultiArray # type: ignore # noqa: E261
from rclpy.qos import qos_profile_sensor_data # type: ignore # noqa: E261

import carb
import omni
from isaacsim.core.utils.stage import get_current_stage
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from bernard_rl.tasks.manager_based.bernard_rl.locomotion_env_cfg import BernardLocomotionEnvCfg_PLAY

import numpy as np

TASK = "Isaac-Bernard-Locomotion-Play-v0"
RL_LIBRARY = "rsl_rl"
DEFAULT_CHECKPOINT = "/home/blaszar/bernard_rl/scripts/rsl_rl/logs/rsl_rl/bernard_locomotion/2025-09-10_11-25-57/model_6100.pt"


class BernardRosNode(Node):
    def __init__(self):
        super().__init__("bernard_isaacsim_hil_node")

        # publisher IMU
        self.pub_imu = self.create_publisher(Imu, "isaacsim_hil/imu", qos_profile_sensor_data)
        # publisher joint states
        self.pub_joints = self.create_publisher(JointState, "isaacsim_hil/joint_states", qos_profile_sensor_data)
        # publisher feet pressure
        self.pub_feet = self.create_publisher(Float32MultiArray, "isaacsim_hil/feet_pressure", qos_profile_sensor_data)
        # publisher gait phase
        self.pub_gait = self.create_publisher(Float32MultiArray, "isaacsim_hil/gait_phase", qos_profile_sensor_data)

        self.sub_actions = self.create_subscription(
            Float32MultiArray,
            "actions",
            self.actions_callback,
            10,
        )
        self.received_actions = None
        self.new_action = False

    def publish_imu(self, lin_accel, ang_vel, orientation):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.linear_acceleration.x = float(lin_accel[0])
        msg.linear_acceleration.y = float(lin_accel[1])
        msg.linear_acceleration.z = float(lin_accel[2])
        msg.angular_velocity.x = float(ang_vel[0])
        msg.angular_velocity.y = float(ang_vel[1])
        msg.angular_velocity.z = float(ang_vel[2])
        msg.orientation.x = float(orientation[0])
        msg.orientation.y = float(orientation[1])
        msg.orientation.z = float(orientation[2])
        msg.orientation.w = float(orientation[3])
        self.pub_imu.publish(msg)

    def publish_joint_states(self, names, positions, velocities):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = [float(p) for p in positions]
        msg.velocity = [float(v) for v in velocities]
        self.pub_joints.publish(msg)

    def publish_feet_pressure(self, pressures):
        msg = Float32MultiArray()
        msg.data = [float(p) for p in pressures]
        self.pub_feet.publish(msg)
    
    def publish_gait_phase(self, gait_phase):
        msg = Float32MultiArray()
        msg.data = [float(g) for g in gait_phase]
        self.pub_gait.publish(msg)
    
    def actions_callback(self, msg):
        self.received_actions = torch.tensor(msg.data, dtype=torch.float32)
        self.new_action = True


class BernardLocomotionDemo:
    """This class provides an interactive demo for the Bernard locomotion environment.
    It loads a provided checkpoint for the Isaac-Bernard-Locomotion-Play-v0 task, trained with RSL RL
    and defines a set of keyboard commands for directing motion of selected robots.

    A robot can be selected from the scene through a mouse click. Once selected, the following
    keyboard controls can be used to control the robot:

    * UP: go forward
    * LEFT: turn left
    * RIGHT: turn right
    * DOWN: stop
    * C: switch between third-person and perspective views
    * ESC: exit current third-person view"""

    def __init__(self):
        """Initializes environment config designed for the interactive model and sets up the environment,
        loads pre-trained checkpoints, and registers keyboard events."""
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        # load the trained jit policy
        if args_cli.checkpoint is not None:
            model_path = retrieve_file_path(args_cli.checkpoint)
        else:
            model_path = DEFAULT_CHECKPOINT
        # create environment
        env_cfg = BernardLocomotionEnvCfg_PLAY()
        env_cfg.scene.num_envs = 25
        env_cfg.episode_length_s = 1000000
        env_cfg.curriculum = None
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.45)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.heading = (-1.0, 1.0)
        # wrap around environment for rsl-rl
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        # load previously trained model
        ppo_runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        ppo_runner.load(model_path)
        # obtain the trained policy for inference
        self.policy = ppo_runner.get_inference_policy(device=self.device)

        self.create_camera()
        self.commands = torch.zeros(env_cfg.scene.num_envs, 3, device=self.device)
        self.commands[:, :] = self.env.unwrapped.command_manager.get_command("base_velocity")
        self.set_up_keyboard()
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)

    def create_camera(self):
        """Creates a camera to be used for third-person view."""
        stage = get_current_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        # Create camera
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.perspective_path)

    def set_up_keyboard(self):
        """Sets up interface for keyboard input and registers the desired keys for control."""
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        T = 0.45
        R = 0.3
        self._key_to_control = {
            "W": torch.tensor([T, 0.0, 0.0], device=self.device),
            "S": torch.tensor([-T, 0.0, 0.0], device=self.device),
            "A": torch.tensor([0.0, T, 0.0], device=self.device),
            "D": torch.tensor([0.0, -T, 0.0], device=self.device),
            "Q": torch.tensor([0.0, 0.0, -R], device=self.device),
            "E": torch.tensor([0.0, 0.0, R], device=self.device),
            "ZEROS": torch.tensor([0.0, 0.0, 0.0], device=self.device),
        }

    def _on_keyboard_event(self, event):
        """Checks for a keyboard event and assign the corresponding command control depending on key pressed."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Arrow keys map to pre-defined command vectors to control navigation of robot
            if event.input.name in self._key_to_control:
                if self._selected_id:
                    self.commands[self._selected_id] = self._key_to_control[event.input.name]
                    print(f"Command for robot {self._selected_id} set to {self.commands[self._selected_id]}")
            # Escape key exits out of the current selected robot view
            elif event.input.name == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
            # C key swaps between third-person and perspective views
            elif event.input.name == "C":
                if self._selected_id is not None:
                    if self.viewport.get_active_camera() == self.camera_path:
                        self.viewport.set_active_camera(self.perspective_path)
                    else:
                        self.viewport.set_active_camera(self.camera_path)
        # On key release, the robot stops moving
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if self._selected_id:
                self.commands[self._selected_id] = self._key_to_control["ZEROS"]

    def update_selected_object(self):
        """Determines which robot is currently selected and whether it is a valid Bernard robot.
        For valid robots, we enter the third-person view for that robot.
        When a new robot is selected, we reset the command of the previously selected
        to continue random commands."""

        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            self._selected_id = None
            self.viewport.set_active_camera(self.perspective_path)
        elif len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please only select one!")
        else:
            prim_splitted_path = selected_prim_paths[0].split("/")
            # a valid robot was selected, update the camera to go into third-person view
            if len(prim_splitted_path) >= 4 and prim_splitted_path[3][0:4] == "env_":
                self._selected_id = int(prim_splitted_path[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("The selected prim was not a H1 robot")

        # Reset commands for previously selected robot if a new one is selected
        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            self.env.unwrapped.command_manager.reset([self._previous_selected_id])
            self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")

    def _update_camera(self):
        """Updates the per-frame transform of the third-person view camera to follow
        the selected robot's torso transform."""

        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[self._selected_id, :]  # - env.scene.env_origins
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[self._selected_id, :]

        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos

        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)


def main():
    """Main function."""
    rclpy.init()
    ros_node = BernardRosNode()

    demo = BernardLocomotionDemo()
    obs, _ = demo.env.reset()

    # Setup logging for 1000 steps
    log_ref_actions = []
    log_bernard_actions = []
    log_obs0 = []
    max_log_steps = 1000
    logs_saved = False

    while simulation_app.is_running():
        # check for selected robots
        demo.update_selected_object()

        obs_iterable = demo.env.unwrapped.observation_manager.get_active_iterable_terms(0)
        for name, raw_value in obs_iterable:
            if name == "policy-base_lin_acc":
                imu_lin_acc = raw_value
            if name == "policy-base_ang_vel":
                imu_ang_vel = raw_value
            if name == "policy-base_orientation":
                imu_orientation = raw_value
            if name == "policy-joint_pos":
                joint_pos = raw_value
            if name == "policy-joint_vel":
                joint_vel = raw_value
            if name == "policy-contact_forces":
                contact_forces = raw_value
            if name == "policy-gait_phase":
                gait_phase = raw_value

        ros_node.publish_imu(imu_lin_acc, imu_ang_vel, imu_orientation)
        ros_node.publish_joint_states([f"joint_{i}" for i in range(6)], joint_pos, joint_vel)
        ros_node.publish_gait_phase(gait_phase)
        ros_node.publish_feet_pressure(contact_forces)

        while not ros_node.new_action:
            rclpy.spin_once(ros_node, timeout_sec=0.1)
        ros_node.new_action = False

        # 1D action tensor for env 0
        bernard_action = ros_node.received_actions.to(demo.device).view(-1)

        # Capture observation for env 0 before the step (used to compute ref_action)
        obs0_to_log = obs[0, :].detach().cpu()

        with torch.inference_mode():
            action = demo.policy(obs)
            ref_action = action[0, :].clone()
            action[0, :] = bernard_action
            obs, _, _, _ = demo.env.step(action)
            # overwrite command based on keyboard input
            obs[:, 27:30] = demo.commands

        # Log actions and save once after 1000 steps
        if len(log_ref_actions) < max_log_steps:
            log_ref_actions.append(ref_action.detach().cpu())
            log_bernard_actions.append(bernard_action.detach().cpu())
            log_obs0.append(obs0_to_log)
            if len(log_ref_actions) == max_log_steps and not logs_saved:
                np.save("ref_actions.npy", torch.stack(log_ref_actions).numpy())
                np.save("bernard_actions.npy", torch.stack(log_bernard_actions).numpy())
                np.save("obs0.npy", torch.stack(log_obs0).numpy())
                logs_saved = True

        rclpy.spin_once(ros_node, timeout_sec=0.0)


if __name__ == "__main__":
    main()
    simulation_app.close()
