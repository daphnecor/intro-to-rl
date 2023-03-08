# -*- coding: utf-8 -*-
""" Prisoner's dilemma environment with OpenAI gym """

from typing import Dict, Tuple

import gym
import numpy as np
from gym import Space, spaces


class PrisonersDilemmaEnv(gym.Env):
    """The standard prisoners dilemma.

    Players simultaneously choose whether to defect (0) or cooperate (1).
    """

    action_space: Space[int]
    observation_space: Space[None | int]
    reward_range: Tuple[float, float]

    def __init__(
        self,
        reward_payoff: int,
        tempta_payoff: int,
        sucker_payoff: int,
        punish_payoff: int,
    ):
        self.observation_space = spaces.Space()
        self.action_space = spaces.Discrete(2)
        self.reward_range = (sucker_payoff, tempta_payoff)
        self.payoff_matrix = self.create_payoff_matrix(
            reward_payoff, tempta_payoff, sucker_payoff, punish_payoff
        )

    def create_payoff_matrix(
        self,
        reward_payoff: int,
        tempta_payoff: int,
        sucker_payoff: int,
        punish_payoff: int,
    ) -> Dict[str, np.ndarray]:
        """Create a payoff dictionary.

        Args:
          reward_payoff (int): reward for mutual cooperation
          temp_payoff (int): temptation to defect
          sucker_payoff (int): when exploited by the other player
          punish_payoff (int): punishment for mutual defection

        Returns:
          Dict[str, Tuple[float, float]]: string actions to rewards dictionary
        """
        payoff_dict = {
            "(D, D)": (punish_payoff, punish_payoff),
            "(D, C)": (tempta_payoff, sucker_payoff),
            "(C, D)": (sucker_payoff, tempta_payoff),
            "(C, C)": (reward_payoff, reward_payoff),
        }
        return payoff_dict

    def _get_reward(
        self, action_agent_one: int, action_agent_two: int
    ) -> Tuple[float, float]:
        """Return rewards for both agents.

        Args:
          action_agent_one (int): player one's action
          action_agent_two (int): player two's action

        Returns:
          Tuple[float, float]: reward for each player
        """
        if action_agent_one == 0 and action_agent_two == 0:
            return self.payoff_matrix["(D, D)"]

        if action_agent_one == 0 and action_agent_two == 1:
            return self.payoff_matrix["(D, C)"]

        if action_agent_one == 1 and action_agent_two == 0:
            return self.payoff_matrix["(C, D)"]

        if action_agent_one == 1 and action_agent_two == 1:
            return self.payoff_matrix["(C, C)"]

        raise NotImplementedError("Actions not known.")

    def step(
        self, action: Tuple[int, int]
    ) -> Tuple[None, Tuple[float, float], bool, bool, Dict]:
        """Interact with the environment.

        Args:
            action (Tuple[int, int]): actions taken by the two players

        Returns:
            None: observation, not used
            Tuple[float, float]: reward for each player
            bool: whether the game is done
            bool: whether the env is truncated, not used
            Dict: information dictionary, not used
        """

        # Get rewards
        rewards = self._get_reward(*action)

        return (
            None,  # observation
            rewards,  # reward
            True,  # done
            False,  # truncated
            {},  # info
        )

    def reset(
        self,
        *,
        seed: None | int = None,  # pylint: disable=unused-argument
        options: None | Dict = None,  # pylint: disable=unused-argument
    ) -> Tuple[None, Dict]:
        """Reset the Prisoner's dilemma environment.

        Args:
            seed (None | int): random seed. Defaults to `None`
            options (None | Dict): reset options. Defaults to `None`

        Returns:
            None: observation, not used
            Dict: information dictionary, not used

        """
        return None, {}
