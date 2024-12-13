from numpy import cos, arccos, sin, pi, sqrt, cross
from numpy.linalg import norm
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
    r1 = norm(r1_vec)
    r2 = norm(r2_vec)
    c12 = cross(r1_vec, r2_vec)
    theta = arccos(r1_vec @ r2_vec / r1 / r2)
    if theta == 0.0:
        raise ValueError('Lambert solution cannot be computed for collinear vectors')
    if prograde and c12[2] <= 0:
        theta = 2 * pi - theta
    elif not prograde and c12[2] >= 0:
        theta = 2 * pi - theta
    A = sin(theta) * sqrt(r1 * r2 / (1 - cos(theta)))
    z = -4 * pi ** 2
    F = -1
    while F < 0:
        z += 0.1
        C = stumpff_c(z)
        S = stumpff_s(z)
        yz = r1 + r2 + A * (z * S - 1) / sqrt(C)
        if yz < 0:
            continue
        F = (yz / C) ** 1.5 * S + A * sqrt(yz) - sqrt(k) * dt
    iter = 0
    ratio = 1
    while abs(ratio) > tol and iter <= max_iter:
        iter += 1
        C = stumpff_c(z)
        S = stumpff_s(z)
        yz = r1 + r2 + A * (z * S - 1) / sqrt(C)
        if yz < 0:
            z = -z / 10  # 避免反复震荡
            continue
        F = (yz / C) ** 1.5 * S + A * sqrt(yz) - sqrt(k) * dt
        if z == 0:
            dF = sqrt(2) / 40 * yz ** 1.5 + A / 8 * (sqrt(yz) + A * sqrt(1 / (2 * yz)))
        else:
            dF = (yz / C) ** 1.5 * ((C - 1.5 * S / C) / (2 * z) + 0.75 * S ** 2 / C) + \
                (A / 8) * ((3 * S * sqrt(yz) / C) + A * sqrt(C / yz))
        ratio = F / dF
        z -= ratio
    f = 1 - yz / r1
    g = A * sqrt(yz / k)
    dg = 1 - yz / r2
    v1_vec = (r2_vec - f * r1_vec) / g
    v2_vec = (dg * r2_vec - r1_vec) / g
    return v1_vec, v2_vec
