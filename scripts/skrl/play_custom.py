# Copyright (c) 2025, Błażej Szargut.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
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
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import skrl
import torch
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
    from bernard_rl.nets.rnn.lstm import Policy, Value
else:
    raise NotImplementedError(
        f"Architecture {args_cli.architecture} not supported. Currently only GRU and LSTM are supported."
    )

if args_cli.ml_framework.startswith("torch"):
    from skrl.agents.torch.ppo import PPO_RNN as PPO
    from skrl.memories.torch import RandomMemory
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.resources.schedulers.torch import KLAdaptiveLR
elif args_cli.ml_framework.startswith("jax"):
    raise NotImplementedError(
        f"ML framework {args_cli.ml_framework} not supported. Currently only torch is supported."
    )

import bernard_rl.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import (
    get_checkpoint_path,
    load_cfg_from_registry,
    parse_env_cfg,
)

# config shortcuts
algorithm = args_cli.algorithm.lower()
architecture = args_cli.architecture.lower()
if architecture in ["gru", "lstm"]:
    agent_cfg_entry_point = "skrl_rnn_ppo_cfg_entry_point"
else:
    raise NotImplementedError(
        f"Architecture {architecture} not supported. Currently only GRU and LSTM are supported."
    )


def main():
    """Play RNN-based RL skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    experiment_cfg = load_cfg_from_registry(args_cli.task, agent_cfg_entry_point)

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (physics) dt for real-time evaluation
    try:
        dt = env.physics_dt
    except AttributeError:
        dt = env.unwrapped.physics_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints

    # instantiate a memory as rollout buffer (any memory can be used for this)
    memory = RandomMemory(memory_size=experiment_cfg["memory"]["memory_size"], num_envs=env.num_envs, device=env.device)

    # instantiate the agent's models (function approximators).
    # PPO requires 2 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
    models = {}
    models["policy"] = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        num_envs=env.num_envs,
        num_layers=experiment_cfg["models"]["policy"]["network"]["num_layers"],
        hidden_size=experiment_cfg["models"]["policy"]["network"]["hidden_size"],
        sequence_length=experiment_cfg["models"]["policy"]["network"]["sequence_length"],
        reduction=experiment_cfg["models"]["policy"]["network"]["reduction"],
        min_log_std=experiment_cfg["models"]["policy"]["min_log_std"],
        max_log_std=experiment_cfg["models"]["policy"]["max_log_std"],
        clip_actions=experiment_cfg["models"]["policy"]["clip_actions"],
        clip_log_std=experiment_cfg["models"]["policy"]["clip_log_std"],
    )
    models["value"] = Value(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        num_envs=env.num_envs,
        num_layers=experiment_cfg["models"]["value"]["network"]["num_layers"],
        hidden_size=experiment_cfg["models"]["value"]["network"]["hidden_size"],
        sequence_length=experiment_cfg["models"]["value"]["network"]["sequence_length"],
        clip_actions=experiment_cfg["models"]["value"]["clip_actions"],
    )

    # configure preprocessors
    if experiment_cfg["agent"]["state_preprocessor"] == "RunningStandardScaler":
        experiment_cfg["agent"]["state_preprocessor_kwargs"] = {
            "size": env.observation_space,
            "device": env.device,
        }
        experiment_cfg["agent"]["state_preprocessor"] = RunningStandardScaler

    if experiment_cfg["agent"]["value_preprocessor"] == "RunningStandardScaler":
        experiment_cfg["agent"]["value_preprocessor_kwargs"] = {
            "size": 1,
            "device": env.device,
        }
        experiment_cfg["agent"]["value_preprocessor"] = RunningStandardScaler

    if experiment_cfg["agent"]["learning_rate_scheduler"] == "KLAdaptiveLR":
        experiment_cfg["agent"]["learning_rate_scheduler_kwargs"] = {
            "kl_threshold": experiment_cfg["agent"]["learning_rate_scheduler_kwargs"]["kl_threshold"],
        }
        experiment_cfg["agent"]["learning_rate_scheduler"] = KLAdaptiveLR

    # configure and instantiate the RL agent
    # https://skrl.readthedocs.io/en/latest/intro/examples.html#gymnasium-gym-environments
    agent = PPO(
        models=models,
        memory=memory,
        cfg=experiment_cfg["agent"],
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    agent.init(experiment_cfg["trainer"])
    agent.load(resume_path)
    # set agent to evaluation mode
    agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
