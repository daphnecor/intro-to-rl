# -*- coding: utf-8 -*-
"""Collection of RL algorithms"""

import numpy as np
from gym import Env


class QlearningAgent:
    """
    Standard Q-learning agent that learns the value of the available actions.
    When state=False, the agent does not get observations. Otherwise, actions
    are conditioned on the action of the other agent.
    """

    def __init__(
        self,
        env: Env,
        eps: float,
        alpha: float,
        gamma: float,
        state: bool = False,
        init_method: str = "zeros",
        verbose: bool = False,
    ):
        self.env = env
        self.eps = self._get_eps(eps)
        self.alpha = self._get_step_size(alpha)
        self.gamma = gamma
        self.state = state
        self.q_table = self.init_q_table(init_method, state, env)
        self.verbose = verbose
        self.total_reward = 0

    def init_q_table(self, init_method: str, state: np.ndarray, env: Env) -> np.ndarray:
        """Initialize Q-table."""
        # Determine size of Q-table
        if state:
            q_table_size = (env.action_space.n, env.action_space.n)
        else:
            q_table_size = env.action_space.n

        # Determine initial values
        if init_method == "zeros":
            return np.zeros((q_table_size))
        raise NotImplementedError("Initialization method not known.")

    def get_action(self, obs: None | np.ndarray = None) -> np.ndarray:
        """Agent takes a new action.

        Args:
            obs (np.ndarray): The action of the other agent.
        """
        # Explore
        if np.random.random() < self._get_eps(self.eps):
            action = np.array([self.env.action_space.sample()])

        # Exploit
        else:
            # If state is given, condition on the action of the other agent
            if obs is not None:
                action = np.random.choice(
                    a=np.argwhere(
                        (self.q_table[:, obs] == self.q_table[:, obs].max())
                    ).flatten(),
                    size=(1,),
                )
            else:
                action = np.random.choice(
                    a=np.argwhere((self.q_table == self.q_table.max())).flatten(),
                    size=(1,),
                )

        if self.verbose:
            print(f"action taken: {action}")

        return action

    def learn(self, action: int, reward: int, obs: None | np.ndarray = None) -> None:
        """Update Q-values for each state, action pair.

        Args:
            state (object): the state of the environment
            action (int): action chosen by agent
            reward (int): reward obtained in episode
        """
        self.total_reward += reward

        if obs is not None:
            self.q_table[action, obs] = self.q_table[action, obs] + self.alpha * (
                reward + self.gamma * np.max(self.q_table) - self.q_table[action, obs]
            )
        else:
            self.q_table[action] = self.q_table[action] + self.alpha * (
                reward + self.gamma * np.max(self.q_table) - self.q_table[action]
            )

        if self.verbose:
            print(f"Q-table: \n {self.q_table}")

    def _get_eps(self, eps: float) -> float:
        """Get the exploration prob for an episode."""
        return eps

    def _get_step_size(self, alpha: float) -> float:
        """Get the step size."""
        return alpha
