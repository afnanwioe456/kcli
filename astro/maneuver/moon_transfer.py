from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numba import njit
from astropy import units as u

from ..core.kepler import a2h, r2nu, period
from ..orbit import Orbit
from ..frame import OrbitalFrame, PQWFrame

if TYPE_CHECKING:
    from ..body import *


@njit
def flyby(rp_p, rp_m, rm_vec, vm_vec, GM_p, GM_m, soi, rel_inc):
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

def transfer_target(orbit: Orbit, moon: Body, cap_t, rp_p, rp_m, rel_inc):
    """向卫星转移的瞄准轨道, 位于捕获临界前

    Args:
        moon (Body): 卫星
        cap_t (Quantity): 捕获瞄准时刻
        rp_p (Quantity): 出发轨道近地点
        rp_m (Quantity): 停泊轨道瞄准近地点
        rel_inc (Quantity): 瞄准轨道与卫星轨道面夹角(-pi~pi)

    Returns:
        Orbit: 转移瞄准轨道
    """
    orb_m = moon.orbit.propagate_to_epoch(cap_t)
    rm_vec = orb_m.r_vec.to_value(u.km)
    vm_vec = orb_m.v_vec.to_value(u.km / u.s)
    rp_p = rp_p.to_value(u.km)
    rp_m = rp_m.to_value(u.km)
    GM_p = moon.attractor.k.to_value(u.km ** 3 / u.s ** 2)
    GM_m = moon.k.to_value(u.km ** 3 / u.s ** 2)
    soi = moon.soi.to_value(u.km)
    rel_inc = rel_inc.to_value(u.rad)

    r_rel, v_rel = flyby(rp_p, rp_m, rm_vec, vm_vec, GM_p, GM_m, soi, rel_inc)
    r_rel *= u.km
    v_rel *= (u.km / u.s)
    cap_t = _find_transfer(moon, orbit, r_rel, cap_t)
    orb_m = orb_m.propagate_to_epoch(cap_t)
    ref = OrbitalFrame.from_orbit(orb_m)
    rp_vec = ref.transform_p_to_parent(r_rel)
    vp_vec = ref.transform_v_to_parent(r_rel, v_rel)
    orb_target = Orbit.from_rv(moon.attractor, rp_vec, vp_vec, cap_t)
    return orb_target

def _find_transfer(moon: Body, orbit: Orbit, r_rel, cap_t_guess, tol=1e-8, max_iter=35):
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
        orb_m = orb_m.propagate_to_epoch(cap_t)
        ref = OrbitalFrame.from_orbit(orb_m)
        r_vec = ref.transform_p_to_parent(r_rel)
        r, nu_t = _get_radius(orbit, -r_vec)
        a = (np.linalg.norm(r_vec) + r) / 2
        T = period(a.to_value(u.km), orbit.attractor.k.to_value(u.km ** 3 / u.s ** 2)) * u.s
        trans_t = cap_t - T / 2
        orb_r = orb_v.propagate_to_epoch(trans_t)
        diff = abs(nu_t - orb_r.nu).to_value(u.rad)
        dt = orb_r.delta_t_at_nu(nu_t) - orb_r.delta_t
        step += 1
    return cap_t

def _get_radius(orbit: Orbit, r_vec):
    """求r_vec方向的轨道径长和真近点角"""
    r_vec = PQWFrame.from_orbit(orbit).transform_d_from_parent(r_vec).to_value(u.km)
    nu = np.arctan2(r_vec[1], r_vec[0])  # arctan象限不清
    if nu < 0:
        nu += 2 * np.pi
    nu = nu * u.rad
    return orbit.r_at_nu(nu), nu

def transfer_start(moon: Body, orb_t: Orbit, orb_v: Orbit):
    # 构建初始轨道
    orb_moon = moon.orbit
    r_cap = orb_t.r_vec
    v_cap = orb_t.v_vec
    p = -r_cap / np.linalg.norm(r_cap)
    h_cap = np.cross(r_cap, v_cap)
    w = h_cap / np.linalg.norm(h_cap)
    q = np.cross(w, p)
    ref = PQWFrame.from_vectors(p, q, w)
    orb_moon = orb_moon.propagate_to_epoch(orb_t.epoch)
    r, _ = _get_radius(orb_v, -r_cap)
    a = (np.linalg.norm(r_cap) + r) / 2
    k = orb_t.attractor.k
    T = period(a.to_value(u.km), k.to_value(u.km ** 3 / u.s ** 2)) * u.s
    v = np.sqrt(- k / a + 2 * k / r)
    r_vec = np.array([r.to_value(u.km), 0, 0], dtype=np.float64) * u.km
    v_vec = np.array([0, v.to_value(u.km/u.s), 0], dtype=np.float64) * u.km / u.s
    r_vec = ref.transform_p_to_parent(r_vec)
    v_vec = ref.transform_d_to_parent(v_vec)
    trans_t = orb_t.epoch - T / 2
    orb_trans_start = Orbit.from_rv(moon.attractor, r_vec, v_vec, trans_t)
    return orb_trans_start
