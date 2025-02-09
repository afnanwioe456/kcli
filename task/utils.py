import math
from numba import njit

@njit
def compute_burn_time(delta_v, thrust, isp, m0):
    """计算轨道机动的点火时长"""
    g0 = 9.80665
    mf = m0 * math.exp(-delta_v / (isp * g0))
    mass_flow_rate = thrust / (isp * g0)
    burn_time = (m0 - mf) / mass_flow_rate
    return burn_time