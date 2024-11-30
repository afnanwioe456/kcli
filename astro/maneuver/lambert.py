import numpy as np
from astropy import units as u
from poliastro.maneuver import Maneuver as PoliMnv
from poliastro.twobody import Orbit as PoliOrbit


def opt_lambert_by_grid_search(orbit_V: PoliOrbit, orbit_T: PoliOrbit, iteration=2, resolution=100):
    """网格搜索双脉冲转移lambert问题的最优能量解

    Args:
        orbit_V (poliastro.twobody.Orbit): 初始轨道
        orbit_T (poliastro.twobody.Orbit): 目标轨道
        iteration (int, optional): 迭代次数. Defaults to 2.
        resolution (int, optional): 网格分辨率. Defaults to 100.

    Returns:
        tuple[float, float, Maneuver]: 等待时间, 转移时间, 机动
    """
    best_lam = None
    best_wt = None
    best_tt = None
    best_dv = None
    wait_l = 0
    wait_r = 2 * orbit_V.period.to_value(u.s)
    trans_l = 0.5 * orbit_V.period.to_value(u.s)
    trans_r = 1.5 * orbit_T.period.to_value(u.s)
    
    for _ in range(iteration):
        wait_step = (wait_r - wait_l) / resolution
        trans_step = (trans_r - trans_l) / resolution
        for t1 in np.arange(wait_l, wait_r, resolution):
            wait_time = t1 * u.s
            orbit_S = orbit_V.propagate(wait_time)
            for t2 in np.arange(trans_l, trans_r, resolution):
                transfer_time = t2 * u.s
                orbit_R = orbit_T.propagate(wait_time + transfer_time)
                lambert = PoliMnv.lambert(orbit_S, orbit_R, prograde=True)
                total_dv = lambert.get_total_cost()
                if not best_dv or total_dv < best_dv:
                    best_wt = wait_time.to_value(u.s)
                    best_tt = transfer_time.to_value(u.s)
                    best_dv = total_dv
                    best_lam = lambert
        wait_l = best_wt - wait_step
        wait_r = best_wt + wait_step
        trans_l = best_tt - trans_step
        trans_r = best_tt + trans_step
        resolution /= 2

    return best_wt, best_tt, best_lam