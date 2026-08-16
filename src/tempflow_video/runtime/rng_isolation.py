from __future__ import annotations

import contextlib
import random
from collections.abc import Iterator

import numpy as np
import torch


@contextlib.contextmanager
def isolated_rng(seed: int | None = None, devices: list[int] | None = None) -> Iterator[None]:
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=devices or [], enabled=True):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

