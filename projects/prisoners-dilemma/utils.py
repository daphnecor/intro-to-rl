# -*- coding: utf-8 -*-
"""Collection of utility functions for the prisoner's dilemma."""
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from prisoners_dilemma.env import PrisonersDilemmaEnv


def vis_action_matrix(indiv_action_seq: np.ndarray, num_agents: int = 2) -> np.ndarray:
    """Get a matrix with the combined actions of both agents

    Args:
        indiv_action_seq (ndarray): action taken for each episode
        num_agents (int): number of agents in the game, defaults to 2

    Returns:
        ndarray: action matrix of size (num_agents * num_actions, num_episodes)
    """

    act_matrix = np.zeros(
        (indiv_action_seq.shape[0], indiv_action_seq.shape[1] * num_agents)
    )
    for i in range(indiv_action_seq.shape[0]):
        # (D, D)
        if indiv_action_seq[i, 0] == 0 and indiv_action_seq[i, 1] == 0:
            act_matrix[i, 0] = 1
        # (D, C)
        elif indiv_action_seq[i, 0] == 0 and indiv_action_seq[i, 1] == 1:
            act_matrix[i, 1] = 1
        # (C, D)
        elif indiv_action_seq[i, 0] == 1 and indiv_action_seq[i, 1] == 0:
            act_matrix[i, 2] = 1
        # (C, C)
        elif indiv_action_seq[i, 0] == 1 and indiv_action_seq[i, 1] == 1:
            act_matrix[i, 3] = 1

    return act_matrix.T


def init_q_tables(init_type: str, env: PrisonersDilemmaEnv) -> np.ndarray:
    """Initialize the Q-tables.

    Args:
        init_type (str): initialization version
        env (PrisonersDilemmaEnv): prisoner's dilemme environment

    Returns:
        ndarray: table of Q-values for the players
    """
    if init_type == "zeros":
        return np.zeros(env.action_space.n), np.zeros(env.action_space.n)
    # TODO: add more ways to initialize the Q-tables

    raise NotImplementedError("Initialization method not known.")


def run_standard_ipd_exp(
    config: Dict,
) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Run a number of in the standard IPD.

    Note: There is only one state and agents cannote observe the others' actions.

    Args:
        config (Dict): game configurations

    Returns:
        ndarray: q-value trajectory of player one
        ndarray: q-value trajectory of player two
        ndarray: sequence of rewards obtained by the players
        ndarray: sequence of actions played by the players
    """
    game_env = PrisonersDilemmaEnv(
        config["payoffs"]["reward_payoff"],
        config["payoffs"]["tempta_payoff"],
        config["payoffs"]["sucker_payoff"],
        config["payoffs"]["punish_payoff"],
    )

    # Initialize Q-tables
    q_table_one, q_table_two = init_q_tables(
        init_type=config["init_type"], env=game_env
    )
    q_traj_one = np.zeros((config["num_episodes"], game_env.action_space.n))
    q_traj_two = np.zeros((config["num_episodes"], game_env.action_space.n))
    rewards_seq = np.zeros((config["num_episodes"], config["num_agents"]))
    action_seq = np.zeros((config["num_episodes"], config["num_agents"]))

    for episode_i in range(config["num_episodes"]):
        if config["verbose"]:
            print(f"Episode: {episode_i}")

        # Player one: take random action
        if np.random.random() < config["params"]["eps"][0]:
            act_play_one = np.array([game_env.action_space.sample()])
        else:  # Exploit
            act_play_one = np.random.choice(
                a=np.argwhere((q_table_one == q_table_one.max())).flatten(),
                size=(1,),
            )
        # Player two: take random action
        if np.random.random() < config["params"]["eps"][1]:
            act_play_two = np.array([game_env.action_space.sample()])
        else:  # Exploit
            act_play_two = np.random.choice(
                a=np.argwhere((q_table_two == q_table_two.max())).flatten(),
                size=(1,),
            )
        # Take a step
        actions = np.concatenate([act_play_one, act_play_two])
        _, rewards, _, _, _ = game_env.step(action=actions)

        # Update Q-values
        q_table_one[act_play_one] = q_table_one[act_play_one] + config["params"][
            "alpha"
        ][0] * (
            rewards[0]
            + config["params"]["gamma"][0] * np.max(q_table_one)
            - q_table_one[act_play_one]
        )

        q_table_two[act_play_two] = q_table_two[act_play_two] + config["params"][
            "alpha"
        ][1] * (
            rewards[1]
            + config["params"]["gamma"][1] * np.max(q_table_two)
            - q_table_two[act_play_two]
        )

        # Store trajectory
        rewards_seq[episode_i, :] = rewards
        action_seq[episode_i, :] = actions
        # episode x actions x players
        q_traj_one[episode_i] = q_table_one
        q_traj_two[episode_i] = q_table_two

    return (
        q_traj_one,
        q_traj_two,
        rewards_seq,
        action_seq,
    )


def run_extended_ipd_exp(config: Dict) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Allow agent two to see the action of agent one.
    """

    game_env = PrisonersDilemmaEnv(
        config["payoffs"]["reward_payoff"],
        config["payoffs"]["tempta_payoff"],
        config["payoffs"]["sucker_payoff"],
        config["payoffs"]["punish_payoff"],
    )

    q_table_one = np.zeros((game_env.action_space.n))
    # Initialize Q-tables (own act x other play act)
    q_table_two = np.zeros((game_env.action_space.n, game_env.action_space.n))

    q_traj_one = np.zeros((config["num_episodes"], game_env.action_space.n))
    q_traj_two = np.zeros(
        (config["num_episodes"], game_env.action_space.n, game_env.action_space.n)
    )
    rewards_seq = np.zeros((config["num_episodes"], config["num_agents"]))
    action_seq = np.zeros((config["num_episodes"], config["num_agents"]), dtype=int)

    # Condition the action of one agent on the action of the other
    for episode_i in range(config["num_episodes"]):
        # # # # Select action player one # # # #
        if np.random.random() < config["params"]["eps"][0]:
            act_play_one = np.array([game_env.action_space.sample()])
        else:  # Exploit
            act_play_one = np.random.choice(
                a=np.argwhere((q_table_one == q_table_one.max())).flatten(), size=(1,)
            )
        # # # # Select action player two | Condition on the action of player one # # # #
        if np.random.random() < config["params"]["eps"][1]:
            act_play_two = np.array([game_env.action_space.sample()])
        else:  # Exploit: choose the action associated with the highest q-value,
            # condition on the action of agent one
            act_play_two = np.random.choice(
                a=np.argwhere(
                    (q_table_two[:, act_play_one] == q_table_two[:, act_play_one].max())
                ).flatten(),
                size=(1,),
            )

        # # # # Take a step # # # #
        actions = np.concatenate([act_play_one, act_play_two])
        _, rewards, _, _, _ = game_env.step(action=actions)

        # # # # Update Q-values # # # #
        q_table_one[act_play_one] = q_table_one[act_play_one] + config["params"][
            "alpha"
        ][0] * (
            rewards[0]
            + config["params"]["gamma"][0] * np.max(q_table_one)
            - q_table_one[act_play_one]
        )

        q_table_two[act_play_two, act_play_one] = q_table_two[
            act_play_two, act_play_one
        ] + config["params"]["alpha"][1] * (
            rewards[1]
            + config["params"]["gamma"][1] * np.max(q_table_two[:, act_play_one])
            - q_table_two[act_play_two, act_play_one]
        )

        # Store trajectory
        rewards_seq[episode_i, :] = rewards
        action_seq[episode_i, :] = actions
        # episode x actions x players
        q_traj_one[episode_i, :] = q_table_one
        q_traj_two[episode_i, :, :] = q_table_two

    return (
        q_traj_one,
        q_traj_two,
        rewards_seq,
        action_seq,
    )


def make_q_vals_fig_standard(
    action_seq: np.ndarray,
    config: Dict,
    q_traj_one: np.ndarray,
    q_traj_two: np.ndarray,
    input_axs: None | np.ndarray = None,
) -> None:
    """
    Visualize the Q-values and actions over training episodes.

    Args:
        action_seq (ndarray): sequence of actions played by the players
        config (Dict): configurations of the game
        q_traj_one (ndarray): Q-value trajectory of player one
        q_traj_two (ndarray): Q-value trajectory of player two
        input_axs (ndarray): axes to use for the figures or None to create a new figure,
            default to `None`
    """
    if input_axs is None:  # Make a single figure
        _, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1]})
        axs[0].set_title(
            f'ϵ=({config["params"]["eps"][0]}, {config["params"]["eps"][1]}),'
            f' γ=({config["params"]["gamma"][0]}, {config["params"]["gamma"][1]})'
        )
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Q-value")

    # Agent one
    agent_one_labels = ["$p_1: Q_D$", "$p_1: Q_C$"]
    agent_one_colors = ["b", "g"]
    for action_i in range(config["num_actions"]):
        axs[0].plot(
            q_traj_one[:, action_i],
            color=agent_one_colors[action_i],
            alpha=0.8,
        )
        axs[0].text(
            q_traj_one.shape[0],
            q_traj_one[-1, action_i],
            agent_one_labels[action_i],
            color=agent_one_colors[action_i],
            fontsize=15,
            weight="bold",
            va="bottom",
        )

    # Agent two
    agent_two_labels = ["$p_2: Q_D$", "$p_2: Q_C$"]
    agent_two_colors = ["r", "orange"]
    for action_i in range(config["num_actions"]):
        axs[0].plot(
            q_traj_two[:, action_i],
            color=agent_two_colors[action_i],
            alpha=0.8,
        )
        axs[0].text(
            q_traj_two.shape[0],
            q_traj_two[-1, action_i],
            agent_two_labels[action_i],
            color=agent_two_colors[action_i],
            fontsize=15,
            weight="bold",
            va="top",
        )

    # Add lines between episodes for if few episodes, otherwise leave out
    line_widths = 0.15 if config["num_episodes"] < 200 else 0

    # Obtain combined action matrix
    comb_act = vis_action_matrix(action_seq)

    sns.heatmap(
        comb_act,
        annot=False,
        cbar=False,
        cmap=["w", "darkgreen"],
        vmin=0,
        vmax=1,
        yticklabels=["(D, D)", "(D, C)", "(C, D)", "(C, C)"],
        xticklabels=[],
        linewidths=line_widths,
        linecolor="k",
        ax=axs[1],
    )
    axs[1].set_title("Actions")
    axs[1].set_xlabel("Episode")
    sns.despine()

    if input_axs is None:
        plt.tight_layout()
        plt.show()
