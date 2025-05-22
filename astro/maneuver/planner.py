import numpy as np
import warnings
from astropy import units as u

from ..core.kepler import *
from ..core.bond import bond
from ..core.izzo import izzo
from ..orbit import Orbit
from ..frame import PQWFrame
from ...math import (vector as mv, rotation as mr)


def apoapsis_planner(orbit: Orbit, ap: u.Quantity, immediate: bool = False):
    if orbit.e >= 1 * u.one and orbit.nu < np.pi * u.rad and not immediate:
        raise ValueError('Cannot handle near-escape orbit with immediate=False')
    ra = ap + orbit.attractor.r

    if not immediate:
        rp = orbit.rp
        orbit_pe = orbit.propagate_to_nu(0 * u.rad, prograde=True)
        v_vec = orbit_pe.v_vec
        dt = orbit_pe.epoch - orbit.epoch
        v_dir = mv.normalize(v_vec)
    else:
        rp = orbit.r
        v_vec = orbit.v_vec
        dt = 0 * u.s
        v_dir = mv.normalize(np.cross(orbit.h_vec, orbit.r_vec))

    a = (rp + ra) / 2
    e = (ra - rp) / (ra + rp)
    h = a2h(
        a.to_value(u.km),
        orbit.attractor.mu.to_value(u.km ** 3 / u.s ** 2),
        e.to_value(u.one)
    ) * u.km ** 2 / u.s
    v1 = h / rp
    imp = v1 * v_dir - v_vec
    return [(dt, imp)]

def periapsis_planner(orbit: Orbit, pe: u.Quantity, immediate: bool = False):
    if orbit.e >= 1 * u.one and not immediate: 
        raise ValueError('Cannot handle near-escape orbit with immediate=False')
    rp = pe + orbit.attractor.r

    if not immediate:
        ra = orbit.ra
        orbit_ap = orbit.propagate_to_nu(np.pi * u.rad, prograde=True)
        v_vec = orbit_ap.v_vec
        dt = orbit_ap.epoch - orbit.epoch
        v_dir = mv.normalize(v_vec)
    else:
        ra = orbit.r
        v_vec = orbit.v_vec
        dt = 0 * u.s
        v_dir = mv.normalize(np.cross(orbit.h_vec, orbit.r_vec))

    a = (rp + ra) / 2
    e = (ra - rp) / (ra + rp)
    h = a2h(
        a.to_value(u.km),
        orbit.attractor.mu.to_value(u.km ** 3 / u.s ** 2),
        e.to_value(u.one)
    ) * u.km ** 2 / u.s
    v1 = h / ra
    imp = v1 * v_dir - v_vec
    return [(dt, imp)]

def match_plane_planner(orb_v: Orbit, orb_t: Orbit, closest: bool = False, conserved: bool = True):
    if orb_v.e >= 1 * u.one: 
        raise ValueError('Cannot handle escape orbit')
    h_vec_v = orb_v.h_vec.to_value(u.km ** 2 / u.s)
    h_vec_t = orb_t.h_vec.to_value(u.km ** 2 / u.s)
    r_vec_v = orb_v.r_vec.to_value(u.km)
    nu_v = orb_v.nu.to_value(u.rad)
    lon = np.cross(h_vec_v, h_vec_t)
    dnu_v = mv.angle_between_vectors(r_vec_v, lon, h_vec_v)

    if closest:
        if dnu_v > np.pi:
            dnu_v -= np.pi
    else:
        r1 = orb_v.r_at_nu(nu_v * u.rad)
        r2 = orb_v.r_at_nu((2 * np.pi - nu_v) * u.rad)
        if r1 < r2:
            dnu_v += np.pi
    nu_v = (nu_v + dnu_v) % (2 * np.pi)

    if conserved:
        # 如果不允许改变轨道形状, 将速度矢量旋转到目标轨道面上
        orb_mnv_v = orb_v.propagate_to_nu(nu_v * u.rad, prograde=True)
        r_vec_v = orb_mnv_v.r_vec.to_value(u.km)
        v_vec_v = orb_mnv_v.vt_vec.to_value(u.km / u.s)
        theta = mr.solve_rotation_angle(v_vec_v, h_vec_t, r_vec_v, np.pi / 2)
        theta = min(theta, key=lambda x: abs((x + np.pi) % (2 * np.pi) - np.pi))
        v_vec_new = mr.vec_rotation(v_vec_v, r_vec_v, theta)
    else:
        # 直接消除法向速度
        orb_mnv_v = orb_v.propagate_to_nu(nu_v * u.rad, prograde=True)
        v_vec_v = orb_mnv_v.v_vec.to_value(u.km / u.s)
        h_i = mv.normalize(h_vec_t)
        delta_v = -np.dot(v_vec_v, h_i) * h_i
        v_vec_new = v_vec_v + delta_v

    imp = (v_vec_new - v_vec_v) * u.km / u.s
    dt = orb_mnv_v.epoch - orb_v.epoch
    return [(dt, imp)]

def lambert_planner(orb_v: Orbit, orb_t: Orbit, solver=bond, **kwargs):
    k = orb_v.attractor.mu.to_value(u.km ** 3 / u.s ** 2)
    r1 = orb_v.r_vec.to_value(u.km)
    r2 = orb_t.r_vec.to_value(u.km)
    dt = (orb_t.epoch - orb_v.epoch).to_value(u.s)
    try:
        v1, v2 = solver(k, r1, r2, dt, **kwargs)
    except ValueError as e:
        warnings.warn(f"solver '{solver.__name__}' failed: {e}, retrying with other solver", RuntimeWarning, 2)
        v1, v2 = izzo(k, r1, r2, tof=dt, **kwargs)
    imp1 = v1 * u.km / u.s - orb_v.v_vec
    imp2 = orb_t.v_vec - v2 * u.km / u.s
    imps = [(0 * u.s, imp1), (dt * u.s, imp2)]
    return imps

def change_phase_planner(orb: Orbit, revisit, inner, conserved=True):
    period = orb.period
    dt = revisit - orb.epoch
    limit_T = T(
        (orb.r / 2).to_value(u.km),
        orb.attractor.mu.to_value(u.km ** 3 / u.s ** 2)
        ) * u.s
    M = dt // period
    if inner != True and M == 0:
        return
    if inner == True:
        inter_T = dt / (M + 1)
    else:
        inter_T = dt / M
    if inter_T < limit_T:
        return
    dv = _get_dv_with_new_T(orb, inter_T)
    imp = orb.v_vec / orb.v * dv
    if conserved:
        return [(0 * u.s, imp), (dt, -imp)]
    else:
        return [(0 * u.s, imp)]

def _get_dv_with_new_T(orb: Orbit, T):
    v0 = orb.v
    v1 = T2v(
        T, orb.r.to_value(u.km), 
        orb.attractor.mu.to_value(u.km ** 3 / u.s ** 2)
        ) * u.km / u.s
    return v1 - v0
    