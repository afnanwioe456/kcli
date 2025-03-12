import math
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
