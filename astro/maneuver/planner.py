import numpy as np
import warnings

from ..core.kepler import *
from ..core.bond import bond
from ..core.izzo import izzo
from ..orbit import Orbit
from ...math import (vector as mv, rotation as mr)


def apoapsis_planner(orbit: Orbit, ap: float, immediate: bool = False):
    if orbit.e >= 1 and orbit.nu < np.pi and not immediate:
        raise ValueError('Cannot handle near-escape orbit with immediate=False')
    ra = ap + orbit.attractor.r

    if not immediate:
        rp = orbit.rp
        orbit_pe = orbit.propagate_to_nu(0)
        v_vec = orbit_pe.v_vec
        dt = orbit_pe.epoch - orbit.epoch
        v_dir = mv.normalize(v_vec)
    else:
        rp = orbit.r
        v_vec = orbit.v_vec
        dt = 0
        v_dir = mv.normalize(np.cross(orbit.h_vec, orbit.r_vec))

    a = (rp + ra) / 2
    e = (ra - rp) / (ra + rp)
    h = a2h(a, orbit.attractor.mu, e)
    v1 = h / rp
    imp = v1 * v_dir - v_vec
    return [(dt, imp)]

def periapsis_planner(orbit: Orbit, pe: float, immediate: bool = False):
    if orbit.e >= 1 and not immediate: 
        raise ValueError('Cannot handle near-escape orbit with immediate=False')
    rp = pe + orbit.attractor.r

    if not immediate:
        ra = orbit.ra
        orbit_ap = orbit.propagate_to_nu(np.pi)
        v_vec = orbit_ap.v_vec
        dt = orbit_ap.epoch - orbit.epoch
        v_dir = mv.normalize(v_vec)
    else:
        ra = orbit.r
        v_vec = orbit.v_vec
        dt = 0
        v_dir = mv.normalize(np.cross(orbit.h_vec, orbit.r_vec))

    a = (rp + ra) / 2
    e = (ra - rp) / (ra + rp)
    h = a2h(a, orbit.attractor.mu, e)
    v1 = h / ra
    imp = v1 * v_dir - v_vec
    return [(dt, imp)]

def match_plane_planner(orb_v: Orbit, orb_t: Orbit, closest: bool = False, conserved: bool = True):
    if orb_v.e >= 1: 
        raise ValueError('Cannot handle escape orbit')
    h_vec_v = orb_v.h_vec
    h_vec_t = orb_t.h_vec
    r_vec_v = orb_v.r_vec
    nu = orb_v.nu
    lon = np.cross(h_vec_v, h_vec_t)
    if npl.norm(lon) < 1e-8:
        return []
    dnu = mv.angle_between_vectors(r_vec_v, lon, h_vec_v)
    if 2 * np.pi - dnu < 1e-8:
        dnu = 0

    if closest and dnu > np.pi:
        dnu -= np.pi
    nu = (nu + dnu) % (2 * np.pi)
    if not closest:
        r1 = orb_v.r_at_nu(nu)
        r2 = orb_v.r_at_nu(nu + np.pi)
        if r1 < r2:
            nu += np.pi
    orb_mnv_v = orb_v.propagate_to_nu(nu)

    if conserved:
        # 如果不允许改变轨道形状, 将速度矢量旋转到目标轨道面上
        r_vec_v = orb_mnv_v.r_vec
        v_vec_v = orb_mnv_v.vt_vec
        theta = mr.solve_rotation_angle(v_vec_v, h_vec_t, r_vec_v, np.pi / 2)
        theta = min(theta, key=lambda x: abs((x + np.pi) % (2 * np.pi) - np.pi))
        v_vec_new = mr.vec_rotation(v_vec_v, r_vec_v, theta)
    else:
        # 直接消除法向速度
        v_vec_v = orb_mnv_v.v_vec
        h_i = mv.normalize(h_vec_t)
        delta_v = -np.dot(v_vec_v, h_i) * h_i
        v_vec_new = v_vec_v + delta_v

    imp = v_vec_new - v_vec_v
    dt = orb_mnv_v.epoch - orb_v.epoch
    return [(dt, imp)]

def lambert_planner(orb_v: Orbit, orb_t: Orbit, solver=bond, **kwargs):
    k = orb_v.attractor.mu
    r1 = orb_v.r_vec
    r2 = orb_t.r_vec
    dt = (orb_t.epoch - orb_v.epoch)
    try:
        v1, v2 = solver(k, r1, r2, dt, **kwargs)
    except ValueError as e:
        warnings.warn(f"solver '{solver.__name__}' failed: {e}, retrying with other solver", RuntimeWarning, 2)
        v1, v2 = izzo(k, r1, r2, tof=dt, **kwargs)
    imp1 = v1 - orb_v.v_vec
    imp2 = orb_t.v_vec - v2
    imps = [(0, imp1), (dt, imp2)]
    return imps

def change_phase_planner(orb: Orbit, dt, inner, conserved=True):
    period = orb.period
    M = dt // period
    if not inner and M == 0:
        return
    if inner:
        inter_period = dt / (M + 1)
    else:
        inter_period = dt / M
    v0 = orb.v
    v1 = T2v(inter_period, orb.r, orb.attractor.mu)
    imp = (v1 - v0) * mv.normalize(orb.v_vec)
    if conserved:
        return [(0, imp), (dt, -imp)]
    else:
        return [(0, imp)]
