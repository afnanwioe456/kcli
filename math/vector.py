import numpy as np
import numpy.linalg as npl
from numba import njit


@njit
def normalize(vec):
    return vec / npl.norm(vec)

@njit
def angle_between_vectors(v1, v2, n=None):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = max(min(cos_theta, 1), -1)
    angle = np.arccos(cos_theta)
    if n is not None and np.dot(np.cross(v1, v2), n) < 0:
        angle = 2 * np.pi - angle
    return angle

@njit
def conic_clamp(vec, axis, min_mag, max_mag, max_angle, prograde=False):
    """
    将vec限制在axis为轴的锥形区域内, 超出时投影到锥面上
    结果限制于min_mag与max_mag之间.
    当prograde置于True时, 向量会被镜像到沿轴正向, 大小置于min_mag
    """
    axis_u = normalize(axis)
    mag_par = np.dot(vec, axis_u)
    v_par = mag_par * axis_u
    v_perp = vec - v_par
    mag_perp = npl.norm(v_perp)

    if prograde and mag_par < 0:
        angle = min(np.arctan2(mag_perp, -mag_par), max_angle)
        mag_par_new = min_mag * np.cos(angle)
        mag_perp_new = min_mag * np.sin(angle)
        v_par = mag_par_new * axis_u
        v_perp = v_perp / mag_perp * mag_perp_new
        return v_par + v_perp

    angle = np.arctan2(mag_perp, mag_par)
    if angle > max_angle and angle < np.pi - max_angle:
        mag_perp_new = abs(mag_par) * np.tan(max_angle)
        v_perp = v_perp / mag_perp * mag_perp_new
        vec = v_par + v_perp
    
    v_norm = npl.norm(vec)
    v_norm_new = min(max(v_norm, min_mag), max_mag)
    vec = vec / v_norm * v_norm_new
    return vec