from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.optimize import dual_annealing, root_scalar
from numba import njit
from astropy import units as u

from ..core.kepler import *
from ..core.lagrange import *
from ..orbit import Orbit
from ..frame import OrbitalFrame, PQWFrame

if TYPE_CHECKING:
    from ..body import *

###
# 径向方向转移, 总是从卫星速度径向入射, 时间和能量都不是最优的
# 在轨道参考系下计算
###

@njit
def _moon_flyby_radial(rp_p, rp_m, rm_vec, vm_vec, GM_p, GM_m, soi, rel_inc):
    # 估算剩余速度
    # TODO: 估算转移轨道远地点为月球轨道高度, 不够准确
    rm = np.linalg.norm(rm_vec)
    vm = np.linalg.norm(vm_vec)
    a_p = (rp_p + rm) / 2
    e_p = (a_p - rp_p) / a_p
    h_p = a2h(a_p, GM_p, e_p)
    v_capture = abs(h_p / rm - vm)  # 捕获时的对地速度
    v_remain2 = v_capture ** 2 - 2 * GM_m / soi  # 剩余速度, 能量守恒
    # 飞越轨道
    e_m = 1 + rp_m * v_remain2 / GM_m
    h_m = rp_m * np.sqrt(v_remain2 + 2 * GM_m / rp_m)
    # 瞄准半径
    theta = np.pi - r2nu(soi, h_m, e_m, GM_m) - np.arccos(1 / e_m)
    radius = soi * np.sin(theta)
    r_rel = np.array([soi * np.cos(theta), radius * np.cos(rel_inc), -radius * np.sin(rel_inc)], dtype=np.float64)
    v_v = h_m / soi
    v_r = np.sqrt(v_capture ** 2 - v_v ** 2)
    v_rel = np.array([-v_r * np.cos(theta) - v_v * np.sin(theta), -v_r * np.sin(theta) + v_v * np.cos(theta), 0], dtype=np.float64)
    v_rel = np.array([v_rel[0], v_rel[1] * np.cos(rel_inc), -v_rel[1] * np.sin(rel_inc)], dtype=np.float64)
    return r_rel, v_rel

def transfer_target_radial(orbit: Orbit, moon: Body, cap_t, rp_m, rel_inc):
    """向卫星转移的瞄准轨道, 位于捕获临界前

    Args:
        orbit (Orbit): 航天器轨道
        moon (Body): 卫星
        cap_t (Quantity): 捕获瞄准时刻
        rp_m (Quantity): 停泊轨道瞄准近星点
        rel_inc (Quantity): 瞄准轨道与卫星轨道面夹角(-pi~pi)

    Returns:
        Orbit: 转移瞄准轨道
    """
    orb_m = moon.orbit.propagate_to_epoch(cap_t)
    rm_vec = orb_m.r_vec
    rp_p = orbit.r_at_nu(orbit.nu_at_direction(-rm_vec)).to_value(u.km)
    rm_vec = orb_m.r_vec.to_value(u.km / u.s)
    vm_vec = orb_m.v_vec.to_value(u.km / u.s)
    rp_m = rp_m.to_value(u.km)
    GM_p = moon.attractor.k.to_value(u.km ** 3 / u.s ** 2)
    GM_m = moon.k.to_value(u.km ** 3 / u.s ** 2)
    soi = moon.soi.to_value(u.km)
    rel_inc = rel_inc.to_value(u.rad)

    r_rel, v_rel = _moon_flyby_radial(rp_p, rp_m, rm_vec, vm_vec, GM_p, GM_m, soi, rel_inc)
    r_rel *= u.km
    v_rel *= (u.km / u.s)
    cap_t = _find_transfer_window(moon, orbit, r_rel, cap_t)
    orb_m = orb_m.propagate_to_epoch(cap_t)
    ref = OrbitalFrame.from_orbit(orb_m)
    rp_vec = ref.transform_p_to_parent(r_rel)
    vp_vec = ref.transform_v_to_parent(r_rel, v_rel)
    orb_target = Orbit.from_rv(moon.attractor, rp_vec, vp_vec, cap_t)
    return orb_target

def _find_transfer_window(moon: Body, orbit: Orbit, r_rel, cap_t_guess, tol=1e-8, max_iter=35):
    """寻找合适的转移窗口, 航天器轨道高度<<卫星轨道高度
    从一个合适的猜测捕获时刻开始, 求转移轨道拱线进而得到出发时刻, 
    比较出发时刻转移轨道相位与航天器相位, 将航天器传播到转移轨道相位,
    反复此步骤直到相位差小于精度要求
    """
    orb_m = moon.orbit
    orb_v = orbit
    cap_t = cap_t_guess
    dt = 0 * u.s
    diff = 1
    step = 0
    while diff > tol and step < max_iter:
        cap_t += dt
        # 将r_rel转到固定参考系
        orb_m = orb_m.propagate_to_epoch(cap_t)
        ref = OrbitalFrame.from_orbit(orb_m)
        r_vec = ref.transform_p_to_parent(r_rel)
        r, nu_t = _get_radius(orbit, -r_vec)
        # 估算转移时间
        a = (np.linalg.norm(r_vec) + r) / 2
        T = T(a.to_value(u.km), orbit.attractor.k.to_value(u.km ** 3 / u.s ** 2)) * u.s
        trans_t = cap_t - T / 2
        # 检查航天器与窗口的相位差
        orb_r = orb_v.propagate_to_epoch(trans_t)
        diff = abs(nu_t - orb_r.nu).to_value(u.rad)
        dt = orb_r.delta_t_at_nu(nu_t) - orb_r.delta_t
        step += 1
    return cap_t

def _get_radius(orbit: Orbit, r_vec):
    """求r_vec方向的轨道径长和真近点角"""
    nu = orbit.nu_at_direction(r_vec)
    return orbit.r_at_nu(nu), nu

def transfer_start_radial(moon: Body, orb_t: Orbit, orb_v: Orbit):
    # 构建初始轨道
    orb_moon = moon.orbit
    r_cap = orb_t.r_vec
    v_cap = orb_t.v_vec
    p = -r_cap / np.linalg.norm(r_cap)
    h_cap = np.cross(r_cap, v_cap)
    w = h_cap / np.linalg.norm(h_cap)
    q = np.cross(w, p)
    # 初始轨道PQW参考系
    ref = PQWFrame.from_vectors(p, q, w)
    orb_moon = orb_moon.propagate_to_epoch(orb_t.epoch)
    # 用径长估计转移轨道
    r, _ = _get_radius(orb_v, -r_cap)
    a = (np.linalg.norm(r_cap) + r) / 2
    k = orb_t.attractor.k
    T = T(a.to_value(u.km), k.to_value(u.km ** 3 / u.s ** 2)) * u.s
    v = np.sqrt(- k / a + 2 * k / r)
    r_vec = np.array([r.to_value(u.km), 0, 0], dtype=np.float64) * u.km
    v_vec = np.array([0, v.to_value(u.km/u.s), 0], dtype=np.float64) * u.km / u.s
    r_vec = ref.transform_p_to_parent(r_vec)
    v_vec = ref.transform_d_to_parent(v_vec)
    trans_t = orb_t.epoch - T / 2
    orb_trans_start = Orbit.from_rv(moon.attractor, r_vec, v_vec, trans_t)
    return orb_trans_start

###
# 寻找向卫星目标倾角轨道转移的最优瞄准轨道
# 确定捕获时刻后, 问题转化为minarg(e) -> f(e, raan, argp) = pe
# 即寻找匹配pe的最小e, 仅适用于pe较小值
# 在轨道参考系下的计算
# e-行星 m-卫星
###

@njit
def _flyby_rv(rp_m, e, inc, raan, argp, soi, GMm):
    """根据捕获轨道参数转换为捕获时状态向量"""
    v_esc2 = (e - 1) * GMm / rp_m
    h = rp_m * np.sqrt(v_esc2 + 2 * GMm / rp_m)
    nu = np.arccos((h ** 2 / (GMm * soi) - 1) / e)
    nu = 2 * np.pi - nu
    rcap_vec, vcap_vec = coe2rv(h, e, inc, raan, argp, nu, GMm)
    return rcap_vec, vcap_vec

@njit
def _transfer_rp(raan, argp, e, rp_m, inc, rm_vec, vm_vec, soi, GMm, GMe):
    """给定捕获轨道参数与卫星状态向量, 求转移轨道pe"""
    rcap_vec, vcap_vec = _flyby_rv(rp_m, e, inc, raan, argp, soi, GMm)
    re_vec = rm_vec + rcap_vec
    ve_vec = vm_vec + vcap_vec
    he, ee = rv2he(re_vec, ve_vec, GMe)
    rp_e = nu2r(0, he, ee, GMe)
    return rp_e

@njit
def _call_transfer_rp_raan_argp(paras, *args):
    raan, argp = paras[0], paras[1]
    return _transfer_rp(raan, argp, *args)

def _min_rp_at_e(e, rp_m, inc, rm_vec, vm_vec, soi, GMm, GMe):
    """给定捕获轨道e, 寻找最小转移轨道pe"""
    args = (e, rp_m, inc, rm_vec, vm_vec, soi, GMm, GMe)
    bounds = [(0, np.pi), (0, 2 * np.pi)]
    result = dual_annealing(_call_transfer_rp_raan_argp, bounds, args, 10)
    return result.fun, result.x
    
def _find_e_bisection(rp_e, rp_m, inc, rm_vec, vm_vec, soi, GMm, GMe, tol=1e-8, maxiter=100):
    """二分法寻找匹配转移轨道pe的最小e"""
    args = (rp_m, inc, rm_vec, vm_vec, soi, GMm, GMe)
    step = 0
    x0 = 1
    x1 = 1.5
    fx0 = _min_rp_at_e(x0, *args)
    rp0 = fx0[0] - rp_e
    fx1 = _min_rp_at_e(x1, *args)
    rp1 = fx1[0] - rp_e
    if rp0 * rp1 > 0:
        raise ValueError()
    while abs(rp0) > tol and step < maxiter:
        x2 = (x1 + x0) / 2
        fx2 = _min_rp_at_e(x2, *args)
        rp2 = fx2[0] - rp_e
        if rp2 * rp0 < 0:
            x1 = x2
        else:
            x0, fx0, rp0 = x2, fx2, rp2
        step += 1
    return x0, *fx0[1]

def _rp_diff_at_e(e, rp_e, *args):
    return _min_rp_at_e(e, *args)[0] - rp_e

def _find_e_root(rp_e, *args):
    solution = root_scalar(_rp_diff_at_e, args=(rp_e, *args), method='brentq', bracket=[1, 1.5])
    e = solution.root
    raan, argp = _min_rp_at_e(e, *args)[1]
    return e, raan, argp

def transfer_target(orbit: Orbit, moon: Body, cap_t, rp_m, inc, relative=True):
    """向卫星转移的瞄准轨道, 位于捕获临界前

    Args:
        orbit (Orbit): 航天器轨道
        moon (Body): 卫星
        cap_t (Quantity): 捕获瞄准时刻
        rp_m (Quantity): 停泊轨道瞄准近星点
        inc (Quantity): 轨道倾角
        relative (bool): 使用相对倾角

    Returns:
        Orbit: 转移瞄准轨道
    """
    orb_m = moon.orbit.propagate_to_epoch(cap_t)
    rm_vec = orb_m.r_vec
    rp_e = orbit.r_at_nu(orbit.nu_at_direction(-rm_vec)).to_value(u.km)
    rm_vec = orb_m.r_vec.to_value(u.km)
    vm_vec = orb_m.v_vec.to_value(u.km / u.s)
    rp_m = rp_m.to_value(u.km)
    GM_e = moon.attractor.k.to_value(u.km ** 3 / u.s ** 2)
    GM_m = moon.k.to_value(u.km ** 3 / u.s ** 2)
    soi = moon.soi.to_value(u.km)
    inc = inc.to_value(u.rad)
    if relative:
        ref = OrbitalFrame.from_orbit(orb_m)
        rm_vec = ref.transform_d_from_parent(rm_vec)
        vm_vec = ref.transform_d_from_parent(vm_vec)
    e, raan, argp = _find_e_root(rp_e, rp_m, inc, rm_vec, vm_vec, soi, GM_m, GM_e)
    rcap_vec, vcap_vec = _flyby_rv(rp_m, e, inc, raan, argp, soi, GM_m)
    re_vec = rm_vec + rcap_vec
    ve_vec = vm_vec + vcap_vec
    if relative:
        re_vec = ref.transform_d_to_parent(re_vec)
        ve_vec = ref.transform_d_to_parent(ve_vec)
    orb_t = Orbit.from_rv(moon.attractor, re_vec * u.km, ve_vec * u.km / u.s, cap_t)
    return orb_t

###
# 寻找向卫星目标轨道转移的最优瞄准轨道
# 问题为minarg(e) -> f(delta_nu, e, argp) = pe
# delta_nu为卫星轨道经过的nu
# 在卫星中心惯性参考系下进行计算
###

@njit
def _call_transfer_rp_argp(paras, e, rp_m, raan, inc, rm_vec, vm_vec, soi, GMm, GMe):
    delta_t, argp = paras[0], paras[1]
    rm_vec, vm_vec = rv2rv_delta_t(rm_vec, vm_vec, delta_t, GMe)
    return _transfer_rp(raan, argp, e, rp_m, inc, rm_vec, vm_vec, soi, GMm, GMe)

def _min_rp_at_e_orbit(e, period, *args):
    args = (e, *args)
    bounds = [(0, period / 2), (0, 2 * np.pi)]
    result = dual_annealing(_call_transfer_rp_argp, bounds, args, 100)
    return result.fun, result.x

def _find_e_orbit_bisection(rp_e, *args, tol=1e-8, maxiter=35):
    """二分法寻找匹配转移轨道pe的最小e"""
    step = 0
    x0 = 1
    x1 = 1.5
    fx0 = _min_rp_at_e_orbit(x0, *args)
    rp0 = fx0[0] - rp_e
    fx1 = _min_rp_at_e_orbit(x1, *args)
    rp1 = fx1[0] - rp_e
    if rp0 * rp1 > 0:
        raise ValueError()
    while abs(rp0) > tol and step < maxiter:
        x2 = (x1 + x0) / 2
        fx2 = _min_rp_at_e_orbit(x2, *args)
        rp2 = fx2[0] - rp_e
        if rp2 * rp0 < 0:
            x1 = x2
        else:
            x0, fx0, rp0 = x2, fx2, rp2
        step += 1
    return x0, *fx0[1]

def _rp_diff_at_e_orbit(e, rp_e, *args):
    return _min_rp_at_e_orbit(e, *args)[0] - rp_e

def _find_e_orbit_root(rp_e, *args):
    solution = root_scalar(_rp_diff_at_e_orbit, args=(rp_e, *args), method='brentq', bracket=[1, 1.5])
    e = solution.root
    delta_t, argp = _min_rp_at_e_orbit(e, *args)[1]
    return e, delta_t, argp

def transfer_orbit_target(orb_v: Orbit, orb_t: Orbit, cap_t, timed=False):
    """向卫星目标轨道转移的瞄准轨道, 位于捕获临界前

    Args:
        orb_v (Orbit): 航天器轨道
        orb_t (Orbit): 目标轨道
        cap_t (Quantity): 瞄准捕获时间
        timed (bool): 严格按瞄准时间捕获

    Returns:
        Orbit: 瞄准轨道
    """
    moon = orb_t.attractor
    orb_m = moon.orbit.propagate_to_epoch(cap_t)
    period = orb_m.period.to_value(u.s)
    rm_vec = orb_m.r_vec.to_value(u.km)
    vm_vec = orb_m.v_vec.to_value(u.km / u.s)
    rp_e = orb_v.ra.to_value(u.km)
    rp_m = orb_t.ra.to_value(u.km)
    GM_e = moon.attractor.k.to_value(u.km ** 3 / u.s ** 2)
    GM_m = moon.k.to_value(u.km ** 3 / u.s ** 2)
    soi = moon.soi.to_value(u.km)
    raan = orb_t.raan.to_value(u.rad)
    inc = orb_t.inc.to_value(u.rad)
    args = (period, rp_m, raan, inc, rm_vec, vm_vec, soi, GM_m, GM_e)
    e, delta_t, argp = _find_e_orbit_root(rp_e, *args)
    rcap_vec, vcap_vec = _flyby_rv(rp_m, e, inc, raan, argp, soi, GM_m)
    orb_m = orb_m.propagate(delta_t * u.s)
    re_vec = orb_m.r_vec + rcap_vec * u.km
    ve_vec = orb_m.v_vec + vcap_vec * u.km / u.s
    orb_t = Orbit.from_rv(moon.attractor, re_vec, ve_vec, orb_m.epoch)
    return orb_t