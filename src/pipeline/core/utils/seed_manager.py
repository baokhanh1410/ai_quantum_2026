"""Centralized random seed management utility for reproducibility."""

import random
import logging
import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """Sets random seed across Python random, NumPy, PyTorch (if available), and Stable-Baselines3.

    Args:
        seed: Integer seed value (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.debug(f"PyTorch random seed set to {seed}")
    except ImportError:
        pass

    try:
        from stable_baselines3.common.utils import set_random_seed as sb3_set_seed

        sb3_set_seed(seed)
        logger.debug(f"Stable-Baselines3 random seed set to {seed}")
    except ImportError:
        pass

    logger.info(f"Global random seed synchronized to: {seed}")
