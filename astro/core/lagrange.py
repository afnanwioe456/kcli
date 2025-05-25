import numpy as np
import numpy.linalg as npl
from numba import njit

from .kepler import *
from .stumpff import stumpff_c, stumpff_s


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
    r = npl.norm(r_vec)
    v = npl.norm(v_vec)
    vr = np.dot(r_vec, v_vec) / r
    alpha = rv2alpha(r, v, GM)
    x = rv2uni_nu(r, vr, delta_t, alpha, GM, tol=tol, max_iter=max_iter)
    z = alpha * x ** 2
    C = stumpff_c(z)
    S = stumpff_s(z)
    f = 1 - x ** 2 / r * C
    g = delta_t - 1 / np.sqrt(GM) * x ** 3 * S
    r1_vec = f * r_vec + g * v_vec
    r1 = npl.norm(r1_vec)
    df = np.sqrt(GM) / r1 / r * (z * S - 1) * x
    dg = 1 - x ** 2 / r1 * C
    v1_vec = df * r_vec + dg * v_vec
    return r1_vec, v1_vec
    