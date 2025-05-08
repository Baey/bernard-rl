# Copyright (c) 2025, Błażej Szargut.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--architecture",
    type=str,
    default="GRU",
    choices=["GRU", "LSTM"],
    help="The architecture used for the RL agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
from datetime import datetime

import gymnasium as gym
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.1"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.architecture.startswith("GRU"):
    from bernard_rl.nets.rnn.gru import Policy, Value
elif args_cli.architecture.startswith("LSTM"):
    raise NotImplementedError(
        f"Architecture {args_cli.architecture} not supported. Currently only GRU is supported."
    )

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

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO_RNN as PPO
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.schedulers.torch import KLAdaptiveLR

# config shortcuts
algorithm = args_cli.algorithm.lower()
architecture = args_cli.architecture.lower()
if architecture in ["gru"]:
    agent_cfg_entry_point = "skrl_gru_ppo_cfg_entry_point"
else:
    raise NotImplementedError(
        f"Architecture {architecture} not supported. Currently only GRU is supported."
    )


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train RNN-based RL agent with skrl."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations if args_cli.max_iterations else agent_cfg["trainer"]["timesteps"]

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{architecture}_{args_cli.ml_framework}"
    print(f"Exact experiment name requested from command line {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # instantiate a memory as rollout buffer (any memory can be used for this)
    memory = RandomMemory(memory_size=agent_cfg["memory"]["memory_size"], num_envs=env.num_envs, device=env.device)

    # instantiate the agent's models (function approximators).
    # PPO requires 2 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
    models = {}
    models["policy"] = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        num_envs=env.num_envs,
        num_layers=agent_cfg["models"]["policy"]["network"]["num_layers"],
        hidden_size=agent_cfg["models"]["policy"]["network"]["hidden_size"],
        sequence_length=agent_cfg["models"]["policy"]["network"]["sequence_length"],
        reduction=agent_cfg["models"]["policy"]["network"]["reduction"],
        min_log_std=agent_cfg["models"]["policy"]["min_log_std"],
        max_log_std=agent_cfg["models"]["policy"]["max_log_std"],
        clip_actions=agent_cfg["models"]["policy"]["clip_actions"],
        clip_log_std=agent_cfg["models"]["policy"]["clip_log_std"],
    )
    models["value"] = Value(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        num_envs=env.num_envs,
        num_layers=agent_cfg["models"]["value"]["network"]["num_layers"],
        hidden_size=agent_cfg["models"]["value"]["network"]["hidden_size"],
        sequence_length=agent_cfg["models"]["value"]["network"]["sequence_length"],
        clip_actions=agent_cfg["models"]["value"]["clip_actions"],
    )

    # configure preprocessors
    if agent_cfg["agent"]["state_preprocessor"] == "RunningStandardScaler":
        agent_cfg["agent"]["state_preprocessor_kwargs"] = {
            "size": env.observation_space,
            "device": env.device,
        }
        agent_cfg["agent"]["state_preprocessor"] = RunningStandardScaler

    if agent_cfg["agent"]["value_preprocessor"] == "RunningStandardScaler":
        agent_cfg["agent"]["value_preprocessor_kwargs"] = {
            "size": 1,
            "device": env.device,
        }
        agent_cfg["agent"]["value_preprocessor"] = RunningStandardScaler

    if agent_cfg["agent"]["learning_rate_scheduler"] == "KLAdaptiveLR":
        agent_cfg["agent"]["learning_rate_scheduler_kwargs"] = {
            "kl_threshold": agent_cfg["agent"]["learning_rate_scheduler_kwargs"]["kl_threshold"],
        }
        agent_cfg["agent"]["learning_rate_scheduler"] = KLAdaptiveLR

    # configure and instantiate the RL agent
    # https://skrl.readthedocs.io/en/latest/intro/examples.html#gymnasium-gym-environments
    agent = PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg["agent"],
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    # configure and instantiate the RL trainer
    trainer = SequentialTrainer(cfg=agent_cfg["trainer"], env=env, agents=[agent])

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        trainer.agents.load(resume_path)

    # start training
    trainer.train()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
