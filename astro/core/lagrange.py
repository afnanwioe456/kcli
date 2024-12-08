from numpy import cos, sin, sqrt, dot, cross
from numpy.linalg import norm
from numba import njit
from astropy import units as u

from .kepler import *
from .stumpff import stumpff_c, stumpff_s


@njit
def rv2rv_delta_nu(r_vec, v_vec, delta_nu, GM):
    """拉格朗日系数法由Δ真近点角推导rv矢量

    Args:
        r_vec (array_like): 初始位置矢量, km
        v_vec (array_like): 初始速度矢量, km / s
        delta_nu (float): 真近点角变化量, rad
        GM (float): 引力参数

    return:
        r_vec, v_vec (ndarray[float]): Δ真近点角处的rv矢量, km/s
    """
    h_vec = cross(r_vec, v_vec)
    h = norm(h_vec)
    r = norm(r_vec)
    vr = dot(v_vec, r_vec) / r
    r = h ** 2 / GM / (1 + (h ** 2 / (GM * r) - 1) * cos(delta_nu) - (h * vr / GM * sin(delta_nu)))
    c = GM / h ** 2 * (1 - cos(delta_nu))
    f = 1 - r * c
    df = (GM / h) * ((1 - cos(delta_nu)) / sin(delta_nu)) * (c - (1 / r) - (1 / r))
    g = r * r * sin(delta_nu) / h
    dg = 1 - r * c
    r_vec = f * r_vec + g * v_vec
    v_vec = df * r_vec + dg * v_vec
    return r_vec, v_vec

@njit
def rv2rv_delta_t(r_vec, v_vec, delta_t, GM, tol=1e-8, max_iter=100):
    """拉格朗日系数法由Δt推导rv矢量

    Args:
        r_vec (array_like): 初始位置矢量, km
        v_vec (array_like): 初始速度矢量, km / s
        delta_nu (float): 真近点角变化量, rad
        GM (float): 引力参数
        tol (float): 误差
        max_iter (int): 最大迭代数

    return:
        r_vec, v_vec (ndarray[float]): Δt处的rv矢量, km/s
    """
    r = norm(r_vec)
    v = norm(v_vec)
    vr = dot(r_vec, v_vec) / r
    alpha = rv2alpha(r, v, GM)
    x = rv2uni_nu(r, vr, delta_t, alpha, GM, tol=tol, max_iter=max_iter)
    z = alpha * x ** 2
    C = stumpff_c(z)
    S = stumpff_s(z)
    f = 1 - x ** 2 / r * C
    g = delta_t - 1 / sqrt(GM) * x ** 3 * S
    r1_vec = f * r_vec + g * v_vec
    r1 = norm(r1_vec)
    df = sqrt(GM) / r1 / r * (z * S - 1) * x
    dg = 1 - x ** 2 / r1 * C
    v1_vec = df * r_vec + dg * v_vec
    return r1_vec, v1_vec
    