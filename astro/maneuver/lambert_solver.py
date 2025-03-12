from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.optimize import dual_annealing
from astropy import units as u

from ..core.izzo import izzo

if TYPE_CHECKING:
    from ..orbit import Orbit


def opt_lambert_by_grid_search(orbit_v: Orbit, 
                               orbit_t: Orbit, 
                               safety_check=True, 
                               before=None,
                               iteration=2,
                               resolution=50,
                               **kwargs):
    """网格搜索双脉冲交会lambert问题的最优能量解

    Args:
        orbit_v (Orbit): 初始轨道
        orbit_t (Orbit): 目标初始轨道
        safety_ckeck (bool, optional): 安全检测. Defaults to True.
        before (Quantity, optional): 限制在该时刻之前转移. Defaults to None.
        iteration (int, optional): 迭代次数. Defaults to 2.
        resolution (int, optional): 网格分辨率. Defaults to 25.

    Returns:
        Maneuver: 相对于初始轨道的机动
    """
    from .create import Maneuver
    best_dv = np.inf * u.km / u.s
    best_lam = None
    epoch = orbit_v.epoch.to_value(u.s)
    pv = orbit_v.period.to_value(u.s)
    pt = orbit_t.period.to_value(u.s)
    if before is None:
        before = 10 * pv + epoch
    else:
        before = before.to_value(u.s)
    trans_l = 0.25 * min(pv,pt)
    trans_r = 0.75 * max(pv,pt)
    wait_l = 60
    wait_r = before - trans_r - epoch
    for _ in range(iteration):
        wait_step = (wait_r - wait_l) / resolution
        trans_step = (trans_r - trans_l) / resolution
        for t1 in np.arange(wait_l, wait_r, wait_step):
            start_epoch = t1 + epoch
            orbit_S = orbit_v.propagate_to_epoch(start_epoch * u.s)
            for t2 in np.arange(trans_l, trans_r, trans_step):
                end_epoch = t2 + start_epoch
                orbit_R = orbit_t.propagate_to_epoch(end_epoch * u.s)
                lambert = Maneuver.lambert(orbit_S, orbit_R, **kwargs)
                total_dv = lambert.get_total_cost()
                if total_dv < best_dv:
                    if safety_check and not lambert.is_safe():
                        continue
                    best_wt = t1
                    best_tt = t2
                    best_dv = total_dv
                    best_lam = lambert
        if best_lam is None:
            raise ValueError('No possible solution.')
        wait_l = max(best_wt - wait_step, wait_l)
        wait_r = min(best_wt + wait_step, wait_r)
        trans_l = max(best_tt - trans_step, trans_l)
        trans_r = min(best_tt + trans_step, trans_r)
        resolution = max(5, resolution / 5)
    best_lam = best_lam.change_orbit(orbit_v)
    return best_lam

def opt_lambert_by_sa(orbit_v: Orbit, 
                      orbit_t: Orbit, 
                      safety_check=True, 
                      before=None):
    """模拟退火求双脉冲交会lambert问题的最优能量解

    Args:
        orbit_v (Orbit): 初始轨道
        orbit_t (Orbit): 目标初始轨道
        safety_ckeck (bool, optional): 安全检测. Defaults to True.
        before (Quantity, optional): 限制在该时刻之前转移. Defaults to None.

    Returns:
        Maneuver: 相对于初始轨道的机动
    """
    epoch = orbit_v.epoch.to_value(u.s)
    pv = orbit_v.period.to_value(u.s)
    pt = orbit_t.period.to_value(u.s)
    if before is None:
        before = (pv * pt) / (abs(pv - pt)) + pv + epoch
    else:
        before = before.to_value(u.s)
    trans_l = 0.25 * min(pv,pt)
    trans_r = 0.75 * max(pv,pt)
    wait_l = 60
    wait_r = before - trans_r - epoch
    args = (orbit_v, orbit_t, epoch, safety_check)
    bounds = [(wait_l, wait_r), (trans_l, trans_r)]
    result = dual_annealing(_call_solve_lambert, bounds, args, 50)
    wt, tt = result.x[0], result.x[1]
    lambert = _solve_lambert(wt, tt, orbit_v, orbit_t, epoch)
    return lambert.change_orbit(orbit_v)
    
def _call_solve_lambert(paras, orbit_v, orbit_t, epoch, safety_check):
    t1, t2 = paras[0], paras[1]
    lambert = _solve_lambert(t1, t2, orbit_v, orbit_t, epoch)
    total_dv = lambert.get_total_cost().to_value(u.m / u.s)
    if safety_check and not lambert.is_safe():
        total_dv += 1e16
    return total_dv

def _solve_lambert(t1, t2, orbit_v, orbit_t, epoch):
    from .create import Maneuver
    start_epoch = t1 + epoch
    orbit_S = orbit_v.propagate_to_epoch(start_epoch * u.s)
    end_epoch = t2 + start_epoch
    orbit_R = orbit_t.propagate_to_epoch(end_epoch * u.s)
    lambert = Maneuver.lambert(orbit_S, orbit_R)
    return lambert

def opt_lambert_revolution(orbit_v, orbit_t, safe_check=True):
    """寻找lambert最佳多周转解

    Args:
        orbit_v (Orbit): 初始轨道
        orbit_t (Orbit): 目标轨道

    Returns:
        Maneuver: 双脉冲转移机动
    """
    from .create import Maneuver
    min_dv = np.inf * u.km / u.s
    best_lam = None
    m = 0
    lowpath = False
    while True:
        try:
            lam = Maneuver.lambert(orbit_v, orbit_t, izzo, M=m, prograde=True, lowpath=lowpath)
        except ValueError:
            if lowpath:
                break
            m = 0
            lowpath = True
            continue
        dv = lam.get_total_cost()
        if dv < min_dv:
            if safe_check and not lam.is_safe():
                continue
            min_dv = dv
            best_lam = lam
        m += 1
    return best_lam
    