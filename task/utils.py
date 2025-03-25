import math
import numpy as np
from numba import njit

@njit
def _burn_time(dv, thrust, isp, m0):
    """计算轨道机动的点火时长"""
    g0 = 9.80665
    mf = m0 * math.exp(-dv / (isp * g0))
    mass_flow_rate = thrust / (isp * g0)
    burn_time = (m0 - mf) / mass_flow_rate
    return burn_time


def compute_burn_time(dv, engines, m0):
    t_total, t_isp_total = 0, 0
    for e in engines:
        t = e.engine.max_thrust * e.engine.thrust_limit
        p = e.engine.specific_impulse
        t_total += t
        t_isp_total += t * p
    thrust = t_total
    isp = t_isp_total / thrust
    return _burn_time(dv, thrust, isp, m0)


def _waterfill_sources(A, r):
    A = np.array(A)
    def out_sum(level):
        return np.sum(np.maximum(A - level, 0))

    # 首先依次尝试各个储量水位, 找到可能的区间
    max_level = np.inf
    for level in A:
        _out = out_sum(level)
        if level < max_level and _out < r:
            max_level = level

    _out = out_sum(max_level)
    count = np.count_nonzero(A >= max_level)  # 还可以继续输出的数量
    level = max_level - (r - _out) / count
    return np.maximum(A - level, 0)


def _waterfill_targets(B, cap, r):
    B = np.array(B)
    cap = np.array(cap)
    res = cap - B  # 残差容量
    def in_sum(level):
        return np.sum(np.minimum(res, np.maximum(level - B, 0)))

    min_level = 0
    levels = np.hstack((B, cap))
    for level in levels:
        _in = in_sum(level)
        if level > min_level and _in < r:
            min_level = level
            
    _in = in_sum(min_level)
    count = np.count_nonzero((B <= min_level) & (cap > min_level))
    level = min_level + (r - _in) / count
    return np.minimum(res, np.maximum(level - B, 0))
    

def waterfill_matrix(A, B, cap, r):
    A = np.array(A)
    B = np.array(B)
    cap = np.array(cap)
    total_out = np.sum(A)
    total_in = np.sum(cap - B)
    A_out = None
    B_in = None
    if r <= 0:
        return np.zeros((A.size, B.size))
    if r >= total_out:
        r = total_out
        A_out = A
    if r >= total_in:
        r = total_in
        B_in = cap - B
        A_out = None
    if A_out is None:
        A_out = _waterfill_sources(A, r)
    if B_in is None:
        B_in = _waterfill_targets(B, cap, r)
    return np.outer(A_out, B_in) / r
