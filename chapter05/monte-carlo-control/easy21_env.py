import gym
from gym import spaces
import numpy as np


class CardDealer:
    """
    Class for drawing cards.
    """

    def __init__(self, is_dealer=False):
        self.hand = list()
        self.hand_sum = 0
        self._is_stick = False
        self.is_dealer = is_dealer

    def reset(self):
        self.__init__()

    def hit(self, first=False):
        """Draw a new card

        Args:
            first (bool, optional): Whether this is the first card of the game. Defaults to False.
        """

        # Draw card
        card_val = np.random.randint(1, 10)

        if first or self.is_dealer:
            card_color = "black"
        else:
            card_color = np.random.choice(["black", "red"], p=[1/3, 2/3])

        # Add card to deck
        self.hand_sum += card_val
        self.hand.append(card_val)

    def stick(self):
        """If True, play out the game.
        """
        self._is_stick = True

    def is_busted(self):
        """If True the agent went busted.

        Returns:
            bool: True if not within interval.
        """
        return not (1 <= self.hand_sum <= 21)

    def is_stick(self):
        """If True the agent no longer takes cards.

        Returns:
            bool: True if player went stick.
        """
        return self._is_stick


class Easy21(gym.Env):
    """Environment for the Easy21 game, a simplification of Blackjack in 
    Sutton & Barto Chapter 5: Exercise 5.1. The assignment is taken from 
    David Silver's course of

    The rules are as follows:
    • The game is played with an infinite deck of cards (i.e. cards are sampled with replacement)
    • Each draw from the deck results in a value between 1 and 10 (uniformly distributed) 
        with a colour of red (probability 1/3) or black (probability 2/3).
    • There are no aces or picture (face) cards in this game
    • At the start of the game both the player and the dealer draw one black
        card (fully observed)
    • Each turn the player may either stick or hit
    • If the player hits then she draws another card from the deck
    • If the player sticks she receives no further cards
    • The values of the player’s cards are added (black cards) or subtracted (red cards)
    • If the player’s sum exceeds 21, or becomes less than 1, then she “goes bust” and loses 
        the game (reward -1)
    • If the player sticks then the dealer starts taking turns. The dealer always 
        sticks on any sum of 17 or greater, and hits otherwise. If the dealer goes 
        bust, then the player wins; otherwise, the outcome – win (reward +1), 
        lose (reward -1), or draw (reward 0) – is the player with the largest sum.
    
    Args:
        gym (Object): Use the OpenAI gym framework.
    """

    def __init__(self, dealer_threshold, player_threshold):

        # Observation Space
        # Dealer's first card [1, 10] and player's sum [1, 21]
        self.observation_space = spaces.Tuple((
            spaces.Box(low=1, high=10, shape=(1,)),
            spaces.Box(low=1, high=21, shape=(1,))
        ))

        # Action Space
        #   0 = sticks (no further cards)
        #   1 = hits (draws another card)
        self.action_space = spaces.Discrete(2)

        # Dimension
        # First card dealer, player's sum, num actions
        self.dim = (10, 21, 2)

        # Dealer's hand
        self.dealer = CardDealer(is_dealer=True)
        self.dealer_threshold = dealer_threshold

        # Player's hand
        self.player = CardDealer(is_dealer=False)
        self.player_threshold = player_threshold

    def get_reward(self):
        """Evaluate the game.

        Returns:
            int: Reward of the player
        """

        are_both_stick = self.dealer.is_stick() and self.player.is_stick()
        if self.dealer.is_busted() or \
                (are_both_stick and self.player.hand_sum > self.dealer.hand_sum):
            return 1  # Player wins
        elif self.player.is_busted() or \
                (are_both_stick and self.player.hand_sum < self.dealer.hand_sum):
            return -1  # Dealer wins
        else:
            return 0  # Draw

    def step(self, action):
        """Take a step in the environment.

        Args:
            action (int): Whether to hit (1) or stick (0)

        Returns:
            observation (Tuple): (state, reward, done, info)
        """
        # Stick
        if action == 0 or self.player.hand_sum > self.player_threshold:
            self.player.stick()

            # Play out game
            while not (self.dealer.is_busted() or self.dealer.is_stick()):
                self.dealer.hit()
                if self.dealer.hand_sum >= self.dealer_threshold:
                    self.dealer.stick()

        # Hit
        elif action == 1:
            self.player.hit()

        return self.get_observation()

    def get_observation(self):
        """Create observation.

        Returns:
            (Tuple): (state, reward, done, info)
        """
        return (
            (self.dealer.hand[0], self.player.hand_sum),
            self.get_reward(),
            self.dealer.is_busted() or self.player.is_busted() or
            (self.dealer.is_stick() and self.player.is_stick()),
            self.info
        )

    def _reset(self):
        """Reset the game.

        Returns:
            Game initialization (Tuple): (state, reward, done, info)
        """
        self.player.reset()
        self.player.hit(first=True)
        self.dealer.reset()
        self.dealer.hit(first=True)

        self.info = {
            "dealer_hand": self.dealer.hand,
            "player_hand": self.player.hand,
            "dealer_sum": self.dealer.hand_sum,
            "player_sum": self.player.hand_sum,
            "actions": list(),
            "rewards": list(),
        }
        return self.get_observation()
