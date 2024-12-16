import numpy as np
from numba import njit

@njit
def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("vector's norm cannot be zero")
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = max(min(cos_theta, 1), -1)
    return np.arccos(cos_theta)
