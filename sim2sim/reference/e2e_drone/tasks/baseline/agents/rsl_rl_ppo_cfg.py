# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticRecurrentCfg, RslRlRndCfg
from isaaclab.utils import configclass

@configclass
class QuadcopterPPORunnerCfg(RslRlOnPolicyRunnerCfg):

    curriculum_stage = "static_goals"  # Options: "yaw_alignment", "static_goals". Must set here

    # Curriculum-specific parameters
    if curriculum_stage == "yaw_alignment":
        num_steps_per_env = 75
        gamma = 0.95
        resume = False
    elif curriculum_stage == "static_goals":  # static_goals or default
        num_steps_per_env = 120
        gamma = 0.99
        resume = False

    max_iterations = 100000
    save_interval = 25
    experiment_name = "baseline_direct"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128, 64],
        critic_hidden_dims=[256, 256, 128, 64],
        activation="elu",
    )
    # policy = RslRlPpoActorCriticRecurrentCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[256, 256, 64],
    #     critic_hidden_dims=[256, 256, 64],
    #     activation="elu",
    #     rnn_type="gru",
    #     rnn_hidden_dim=64,
    #     rnn_num_layers=1,
    # )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=200,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=gamma,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # rnd_cfg=RslRlRndCfg(
        #     weight=0.1,
        #     # weight_schedule=RslRlRndCfg.LinearWeightScheduleCfg(
        #     #     final_value=0.02,
        #     #     initial_step=0,
        #     #     final_step=20000,
        #     # ),
        #     reward_normalization=False,
        #     state_normalization=False,
        #     learning_rate=1e-3,
        # )
    )
    resume = resume
    # reset_std = [0.1, 0.1, 0.4, 0.2]
    load_run = "2025-10-13_00-51-18"
    load_checkpoint = "model_2900.pt"
