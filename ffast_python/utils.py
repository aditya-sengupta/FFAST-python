import numpy as np

norm_squared = lambda x: np.inner(x, x)
positive_mod = lambda a, b: (a % b) + b * (a % b < 0)