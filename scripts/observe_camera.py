# observe_camera.py
# View Isaac Lab MDP-camera observation (depth).

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--num_envs", type=int, default=1)
args_cli, hydra_args = parser.parse_known_args()

args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# After sim starts:
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config
import bernard_rl.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

agent_cfg_entry_point = "skrl_cfg_entry_point"



@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    print("[INFO] Creating environment...")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = env.unwrapped


    # simulate physics
    count = 0
    plt.ion()
    fig = plt.figure()
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 5000 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # print current orientation of pole
            img = obs["policy"]["camera"][0].cpu().numpy()
            img = img.squeeze(-1)
            plt.imshow(img, cmap="gray")
            plt.pause(0.001)
            plt.clf()
            # update cou

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
