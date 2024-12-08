from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from astropy import units as u

from ..core.bond import bond

if TYPE_CHECKING:
    from ..orbit import Orbit


def opt_lambert_by_grid_search(orbit_V: Orbit, orbit_T: Orbit, solver=bond, iteration=2, resolution=100, **kwargs):
    """网格搜索双脉冲转移lambert问题的最优能量解

    Args:
        orbit_V (Orbit): 初始轨道
        orbit_T (Orbit): 目标轨道
        iteration (int, optional): 迭代次数. Defaults to 2.
        resolution (int, optional): 网格分辨率. Defaults to 100.

    Returns:
        tuple[float, float, Maneuver]: 等待时间, 转移时间, 机动
    """
    from .create import Maneuver
    best_lam = None
    best_wt = None
    best_tt = None
    best_dv = np.inf * u.km / u.s
    wait_l = 60
    wait_r = 2 * orbit_V.period.to_value(u.s)
    trans_l = 0.5 * orbit_V.period.to_value(u.s)
    trans_r = 1.5 * orbit_T.period.to_value(u.s)
    epoch = orbit_V.epoch
    
    for _ in range(iteration):
        wait_step = (wait_r - wait_l) / resolution
        trans_step = (trans_r - trans_l) / resolution
        for t1 in np.arange(wait_l, wait_r, resolution):
            wait_epoch = t1 * u.s + epoch
            orbit_S = orbit_V.propagate_to_epoch(wait_epoch)
            for t2 in np.arange(trans_l, trans_r, resolution):
                transfer_epoch = t2 * u.s + wait_epoch
                orbit_R = orbit_T.propagate_to_epoch(transfer_epoch)
                lambert = Maneuver.lambert(orbit_S, orbit_R, solver, **kwargs)
                total_dv = lambert.get_total_cost()
                if total_dv < best_dv:
                    best_wt = t1
                    best_tt = t2
                    best_dv = total_dv
                    best_lam = lambert
        wait_l = best_wt - wait_step
        wait_r = best_wt + wait_step
        trans_l = best_tt - trans_step
        trans_r = best_tt + trans_step
        resolution /= 2

    return best_wt, best_tt, best_lam