
from __future__ import annotations

import os
import random

__all__ = ["set_seed", "PAPER_SEEDS"]

PAPER_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

def set_seed(seed: int, *, deterministic_torch: bool = True) -> int:
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
        if cudnn is not None:
            cudnn.deterministic = True
            cudnn.benchmark = False

    return seed
