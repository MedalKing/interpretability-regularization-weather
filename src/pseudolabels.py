from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def make_feature_importance_z(feature_names: Sequence[str]) -> NDArray[np.float32]:
    """
    Construct a simple feature-importance pseudo-label vector z in [0,1]^d.

    The returned z is a heuristic prior about which inputs *should* matter for the
    forecast. In your experiments this z is used as a regularisation target.

    Notes:
      - This is not learned from data; it is author-specified.
      - It is deliberately easy to define "adversarial" pseudo-labels too
        (e.g., making rand_noise the only important feature) for sanity checks.
    """
    z = np.zeros(len(feature_names), dtype=np.float32)

    # Current base feature set:
    # ["tmin", "tavg", "tmax", "prcp", "dayofyear_sin", "dayofyear_cos"]
    # "rand_noise" may be present as an extra feature; it is given
    # a default importance of 0.0 by leaving it out of the mapping below.

    for i, name in enumerate(feature_names):
        if name in {"tmax", "tavg"}:
            z[i] = 1.0
        elif name == "tmin":
            z[i] = 0.7
        elif name == "prcp":
            z[i] = 0.4
        elif name in {"dayofyear_sin", "dayofyear_cos"}:
            z[i] = 0.3

    # Ensure z is in [0,1]
    return np.clip(z, 0.0, 1.0).astype(np.float32)