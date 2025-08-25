# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import mpl_toolkits.mplot3d  # noqa: F401
from scipy.spatial.transform import Rotation as R

try:
    from .motion_loader import MotionLoader
except ImportError:
    from motion_loader import MotionLoader


class MotionViewer:
    def __init__(
        self,
        motion_file: str,
        device: torch.device | str = "cpu",
        render_scene: bool = False,
    ) -> None:
        self._figure = None
        self._figure_axes = None
        self._render_scene = render_scene

        self._motion_loader = MotionLoader(motion_file=motion_file, device=device)

        self._num_frames = self._motion_loader.num_frames
        self._current_frame = 0
        self._body_positions = self._motion_loader.body_positions.cpu().numpy()
        self._dof_positions = self._motion_loader.dof_positions.cpu().numpy()
        self._body_rotations = self._motion_loader.body_rotations.cpu().numpy()
        self._body_linear_velocities = (
            self._motion_loader.body_linear_velocities.cpu().numpy()
        )

        self._colors = plt.cm.jet(np.linspace(0, 1, self._body_positions.shape[1]))  # type: ignore
        self._trajectories = {
            "l_foot": [],
            "r_foot": [],
            "body": [],
        }
        self.l_foot_index = self._motion_loader.get_body_index(["l_foot"])[0]
        self.r_foot_index = self._motion_loader.get_body_index(["r_foot"])[0]
        self.body_index = self._motion_loader.get_body_index(["body"])[0]
        print(
            f"Left foot index: {self.l_foot_index}, Right foot index: {self.r_foot_index}, Body index: {self.body_index}"
        )

        print("\nBody")
        for i, name in enumerate(self._motion_loader.body_names):
            minimum = np.min(self._body_positions[:, i], axis=0).round(decimals=2)
            maximum = np.max(self._body_positions[:, i], axis=0).round(decimals=2)
            print(
                f"  |-- [{name}] minimum position: {minimum}, maximum position: {maximum}"
            )

    def _drawing_callback(self, frame: int) -> None:
        vertices = self._body_positions[self._current_frame]

        self._figure_axes.clear()
        self._figure_axes.scatter(*vertices.T, color=self._colors, depthshade=False)

        for i, name in enumerate(self._motion_loader.body_names):
            if name in ["l_foot", "r_foot", "body"]:
                x, y, z = vertices[i]
                self._figure_axes.text(x, y, z, name, fontsize=16)

        if self._render_scene:
            minimum = np.min(self._body_positions.reshape(-1, 3), axis=0)
            maximum = np.max(self._body_positions.reshape(-1, 3), axis=0)
            center = 0.5 * (maximum + minimum)
            diff = 0.75 * (maximum - minimum)
        else:
            minimum = np.min(vertices, axis=0)
            maximum = np.max(vertices, axis=0)
            center = 0.5 * (maximum + minimum)
            diff = np.array([0.75 * np.max(maximum - minimum).item()] * 3)

        self._figure_axes.set_xlim((center[0] - diff[0], center[0] + diff[0]))
        self._figure_axes.set_ylim((center[1] - diff[1], center[1] + diff[1]))
        self._figure_axes.set_zlim((center[2] - diff[2], center[2] + diff[2]))
        self._figure_axes.set_box_aspect(aspect=diff / diff[0])

        x, y = np.meshgrid(
            [center[0] - diff[0], center[0] + diff[0]],
            [center[1] - diff[1], center[1] + diff[1]],
        )
        self._figure_axes.plot_surface(x, y, np.zeros_like(x), color="green", alpha=0.2)

        label_fontsize = 18
        title_fontsize = 22
        tick_fontsize = 16
        self._figure_axes.set_xlabel("X", fontsize=label_fontsize, labelpad=30)
        self._figure_axes.set_ylabel("Y", fontsize=label_fontsize, labelpad=30)
        self._figure_axes.set_zlabel("Z", fontsize=label_fontsize, labelpad=30)
        self._figure_axes.set_title(f"frame: {self._current_frame}/{self._num_frames}", fontsize=title_fontsize)
        self._figure_axes.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        self._figure_axes.tick_params(axis='both', which='minor', labelsize=tick_fontsize)

        # Trajectories
        self._trajectories["l_foot"].append(np.asarray(vertices[self.l_foot_index]))
        self._trajectories["r_foot"].append(np.asarray(vertices[self.r_foot_index]))
        self._trajectories["body"].append(np.asarray(vertices[self.body_index]))

        for name, traj in self._trajectories.items():
            traj_np = np.array(traj)
            color = (
                "red" if name == "l_foot" else "blue" if name == "r_foot" else "green"
            )
            self._figure_axes.plot(
                traj_np[:, 0], traj_np[:, 1], traj_np[:, 2], color=color, alpha=0.6
            )

        # Velocities
        positions = self._body_positions[self._current_frame]
        velocities = self._body_linear_velocities[self._current_frame]

        # Arrow scaling
        arrow_lengths = np.linalg.norm(velocities, axis=1)
        max_length = np.percentile(arrow_lengths, 95) + 1e-6
        scaled_velocities = velocities * (0.2 / max_length)

        self._figure_axes.quiver(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            scaled_velocities[:, 0],
            scaled_velocities[:, 1],
            scaled_velocities[:, 2],
            color="black",
            alpha=0.5,
        )

        # Quaternion rotations
        orientations = self._body_rotations[self._current_frame]
        positions = self._body_positions[self._current_frame]

        axis_length = 0.1

        for pos, quat in zip([positions[0], positions[8], positions[7]], [orientations[0], orientations[8], orientations[7]]):
            rot = R.from_quat(quat, scalar_first=True)
            axes = rot.apply(np.eye(3) * axis_length)

            # rysujemy 3 strzałki dla każdej osi
            self._figure_axes.quiver(
                *pos, *axes[0], color="r", alpha=0.6
            )  # X - czerwony
            self._figure_axes.quiver(
                *pos, *axes[1], color="g", alpha=0.6
            )  # Y - zielony
            self._figure_axes.quiver(
                *pos, *axes[2], color="b", alpha=0.6
            )  # Z - niebieski

        # Debug prints
        if self._current_frame % 30 == 0:
            dof = self._dof_positions[self._current_frame]
            vel = self._body_linear_velocities[self._current_frame]
            speeds = np.linalg.norm(vel, axis=1)
            print(
                f"[Frame {self._current_frame}] DOF pos: min={dof.min():.2f}, max={dof.max():.2f}"
            )
            print(f"[Frame {self._current_frame}] Mean body speed: {speeds.mean():.3f}")

        self._current_frame += 1
        if self._current_frame >= self._num_frames:
            self._current_frame = self._num_frames - 1
            # self._trajectories = {"l_foot": [], "r_foot": [], "body": []}

    def show(self) -> None:
        self._figure = plt.figure()
        self._figure_axes = self._figure.add_subplot(projection="3d")
        self._animation = matplotlib.animation.FuncAnimation(
            fig=self._figure,
            func=self._drawing_callback,
            frames=self._num_frames,
            interval=1000 * self._motion_loader.dt,
        )
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    parser.add_argument("--render-scene", action="store_true", default=False)
    parser.add_argument("--matplotlib-backend", type=str, default="TkAgg")
    args, _ = parser.parse_known_args()

    matplotlib.use(args.matplotlib_backend)

    viewer = MotionViewer(args.file, render_scene=args.render_scene)
    viewer.show()
