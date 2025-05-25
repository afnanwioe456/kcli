import numpy as np
import numpy.linalg as npl
from numba import njit

from .stumpff import stumpff_c, stumpff_s

@njit
def bond(k, r1_vec, r2_vec, dt, prograde=True, max_iter=100, tol=1e-8):
    """Applies bond algorithm (1996) to solve Lambert problem.

    Args:
        k (float): gravity parameter.
        r1_vec (ndarray): initial position vector.
        r2_vec (ndarray): final position vector.
        dt (float): transfer time.
        prograde (bool, optional): desired inclination of the transfer orbit. Defaults to True.
        max_iter (int, optional): maximum number of iterations. Defaults to 100.
        tol (float, optional): absolute tolerance. Defaults to 1e-8.

    Returns:
        v1_vec, v2_vec: initial and final velocity vector.
    """
    r1 = npl.norm(r1_vec)
    r2 = npl.norm(r2_vec)
    c12 = np.cross(r1_vec, r2_vec)
    theta = np.arccos(r1_vec @ r2_vec / r1 / r2)
    if theta == 0.0:
        raise ValueError('Lambert solution cannot be computed for collinear vectors')
    if prograde and c12[2] <= 0:
        theta = 2 * np.pi - theta
    elif not prograde and c12[2] >= 0:
        theta = 2 * np.pi - theta
    A = np.sin(theta) * np.sqrt(r1 * r2 / (1 - np.cos(theta)))
    z = -4 * np.pi ** 2
    F = -1
    while F < 0:
        z += 0.1
        C = stumpff_c(z)
        S = stumpff_s(z)
        yz = r1 + r2 + A * (z * S - 1) / np.sqrt(C)
        if yz < 0:
            continue
        F = (yz / C) ** 1.5 * S + A * np.sqrt(yz) - np.sqrt(k) * dt
    iter = 0
    ratio = 1
    while abs(ratio) > tol and iter <= max_iter:
        iter += 1
        C = stumpff_c(z)
        S = stumpff_s(z)
        yz = r1 + r2 + A * (z * S - 1) / np.sqrt(C)
        if yz < 0:
            z = -z / 10  # 避免反复震荡
            continue
        F = (yz / C) ** 1.5 * S + A * np.sqrt(yz) - np.sqrt(k) * dt
        if z == 0:
            dF = np.sqrt(2) / 40 * yz ** 1.5 + A / 8 * (np.sqrt(yz) + A * np.sqrt(1 / (2 * yz)))
        else:
            dF = (yz / C) ** 1.5 * ((C - 1.5 * S / C) / (2 * z) + 0.75 * S ** 2 / C) + \
                (A / 8) * ((3 * S * np.sqrt(yz) / C) + A * np.sqrt(C / yz))
        ratio = F / dF
        z -= ratio
    f = 1 - yz / r1
    g = A * np.sqrt(yz / k)
    dg = 1 - yz / r2
    v1_vec = (r2_vec - f * r1_vec) / g
    v2_vec = (dg * r2_vec - r1_vec) / g
    return v1_vec, v2_vec
