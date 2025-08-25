# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoActorCriticQuantCfg


@configclass
class BernardLocomotionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 128
    max_iterations = 10000
    save_interval = 100
    experiment_name = "bernard_locomotion"
    empirical_normalization = True
    # policy = RslRlPpoActorCriticCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[256, 128],
    #     critic_hidden_dims=[256, 128],
    #     activation="relu",
    # )
    # policy = RslRlPpoActorCriticQuantCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[256, 128],
    #     critic_hidden_dims=[256, 128],
    #     activation="qrelu",
    # )
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[128],
        critic_hidden_dims=[128],
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        activation="tanh",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )

