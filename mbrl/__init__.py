import abc

import numpy as np


class Agent:
    @abc.abstractmethod
    def act(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues an action given an observation."""
