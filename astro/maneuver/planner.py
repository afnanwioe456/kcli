import numpy as np
import warnings
from astropy import units as u

from ..core.kepler import *
from ..core.linear import angle_between_vectors
from ..core.bond import bond
from ..core.izzo import izzo
from ..orbit import Orbit
from ..frame import PQWFrame


def apoapsis_planner(orbit: Orbit, ap: u.Quantity):
    if orbit.e >= 1 * u.one and orbit.nu < np.pi:
        raise ValueError('Planner "change_apoasis" cannot handle orbit that is about to escape, try other planner.')
    ra = ap + orbit.attractor.r
    rp = orbit.rp
    a = (rp + ra) / 2
    e = (ra - rp) / (ra + rp)
    h = a2h(
        a.to_value(u.km),
        orbit.attractor.k.to_value(u.km ** 3 / u.s ** 2),
        e.to_value(u.one)
    ) * u.km ** 2 / u.s
    v1 = h / rp
    orbit_pe = orbit.propagate_to_nu(0 * u.rad, prograde=True)
    v0 = orbit_pe.v
    v_vec = orbit_pe.v_vec
    imp = (v1 / v0 - 1) * v_vec
    dt = orbit_pe.epoch - orbit.epoch
    return [(dt, imp)]

def periapsis_planner(orbit: Orbit, pe: u.Quantity):
    if orbit.e >= 1 * u.one: 
        raise ValueError('Planner "change_periapsis" cannot handle escape orbit, try other planner.')
    ra = orbit.ra
    rp = pe + orbit.attractor.r
    a = (rp + ra) / 2
    e = (ra - rp) / (ra + rp)
    h = a2h(
        a.to_value(u.km),
        orbit.attractor.k.to_value(u.km ** 3 / u.s ** 2),
        e.to_value(u.one)
    ) * u.km ** 2 / u.s
    v1 = h / ra
    orbit_ap = orbit.propagate_to_nu(np.pi * u.rad, prograde=True)
    v0 = orbit_ap.v
    v_vec = orbit_ap.v_vec
    imp = (v1 / v0 - 1) * v_vec
    dt = orbit_ap.epoch - orbit.epoch
    return [(dt, imp)]

def match_plane_planner(orb_v: Orbit, orb_t: Orbit, closest: bool = False, conserved: bool = True):
    if orb_v.e >= 1 * u.one: 
        raise ValueError('Planner "match_plane" cannot handle escape orbit, try other planner.')
    hv_vec = orb_v.h_vec.to_value(u.km ** 2 / u.s)
    ht_vec = orb_t.h_vec.to_value(u.km ** 2 / u.s)
    theta = angle_between_vectors(hv_vec, ht_vec)
    if theta < 1e-3:
        return []
    lon = np.cross(hv_vec, ht_vec) * u.km
    nu_v = orb_v.nu_at_direction(lon)
    nu_t = orb_t.nu_at_direction(lon)
    pi_rad = np.pi * u.rad
    if closest:
        delta_nu = nu_v - orb_v.nu
        if delta_nu > pi_rad or delta_nu < -pi_rad:
            nu_v += pi_rad
            nu_t += pi_rad
    else:
        r1 = orb_v.r_at_nu(nu_v)
        r2 = orb_v.r_at_nu(2 * pi_rad - nu_v)
        if r1 < r2:
            nu_v += pi_rad
            nu_t += pi_rad
    orb_mnv_v = orb_v.propagate_to_nu(nu_v, prograde=True)
    orb_mnv_t = orb_t.propagate_to_nu(nu_t)
    vtv = orb_mnv_v.vt
    vtv_vec = orb_mnv_v.vt_vec
    vtt_vec = orb_mnv_t.vt_vec
    vtt_i = vtt_vec / np.linalg.norm(vtt_vec)
    if conserved:
        vtt_vec = vtt_i * vtv
    else:
        vtt_vec = vtt_i * vtv * np.sin(theta)
    imp = vtt_vec - vtv_vec
    dt = orb_mnv_v.epoch - orb_v.epoch
    return [(dt, imp)]

def lambert_planner(orb_v: Orbit, orb_t: Orbit, solver=bond, **kwargs):
    k = orb_v.attractor.k.to_value(u.km ** 3 / u.s ** 2)
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
        orb.attractor.k.to_value(u.km ** 3 / u.s ** 2)
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
    dv = _get_dv(orb, inter_T)
    imp = orb.v_vec / orb.v * dv
    if conserved:
        return [(0 * u.s, imp), (dt, -imp)]
    else:
        return [(0 * u.s, imp)]

def _get_dv(orb: Orbit, T):
    v0 = orb.v
    v1 = T2v(T, orb.r.to_value(u.km), orb.attractor.k.to_value(u.km ** 3 / u.s ** 2)) * u.km / u.s
    return v1 - v0
    