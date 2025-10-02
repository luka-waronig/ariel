"""
RNG for assignment 3.
"""

import numpy as np
import random as rd

SEED = 42
RNG = rd.Random(SEED)
NP_RNG = np.random.default_rng(SEED)
