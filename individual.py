import numpy as np
from typing import NamedTuple

Individual = NamedTuple('Individual', (
    ('weights', np.ndarray),
))
