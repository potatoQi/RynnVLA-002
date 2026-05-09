import logging
from typing import List

from .chameleon import ChameleonConfig

logger = logging.getLogger(__name__)


class ChameleonXLLMXConfig(ChameleonConfig):

    def __init__(
        self,
        z_loss_weight: float = 0.0,
        action_dim: int = 7,
        time_horizon: int = 5,
        transition_token_id: int = 16001,
        transition_token_count: int = 4,
        transition_token_hidden_mult: int = 1,
        **kwargs,
    ):
        self.z_loss_weight = z_loss_weight
        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.transition_token_id = transition_token_id
        self.transition_token_count = transition_token_count
        self.transition_token_hidden_mult = transition_token_hidden_mult
        super().__init__(
            **kwargs,
        )
